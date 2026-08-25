"""
Mini Search Engine - Stage 16 Production Web & API Application
Clean Layered Architecture with Versioned REST API v1, Rate Limiting,
Prometheus Metrics, Health & Readiness Probes, Structured Logging, and Web UI.
"""

import time
import uuid
import logging
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, Response

import config
from services.search_service import SearchService
from services.index_manager import IndexManager
from services.metrics import metrics_registry
from analytics import (
    get_summary_metrics,
    get_top_queries,
    get_top_zero_result_queries,
    get_query_type_distribution,
    get_recent_searches
)
from performance import get_memory_usage
from evaluation.evaluator import SearchEvaluator

# Structured Logging Setup
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger("search_engine")

app = Flask(__name__)
app.config["ENV"] = config.APP_ENV
app.config["DEBUG"] = config.DEBUG

# Initialize Central Services
search_service = SearchService.get_instance()
index_manager = IndexManager()
evaluator = SearchEvaluator()


# --- In-Memory Sliding Window Rate Limiter ---
class RateLimiter:
    def __init__(self, max_requests: int = config.RATE_LIMIT_REQUESTS, window_seconds: int = config.RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        # Clean up old requests outside window
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.window_seconds]
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter()


# --- Middleware: Security Headers, Request ID & Rate Limiting ---
@app.before_request
def before_request_hook():
    request.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
    request.start_time = time.perf_counter()

    client_ip = request.remote_addr or "127.0.0.1"
    if not rate_limiter.is_allowed(client_ip):
        metrics_registry.record_request("rate_limited", 0.0, success=False, error_type="RateLimitExceeded")
        return jsonify({
            "request_id": request.request_id,
            "error": "Too Many Requests: Rate limit exceeded. Please try again later.",
            "status_code": 429
        }), 429


@app.after_request
def add_security_and_cors_headers(response):
    response.headers["X-Request-ID"] = getattr(request, "request_id", uuid.uuid4().hex[:8])
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # CORS Headers
    if "*" in config.ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    else:
        origin = request.headers.get("Origin", "")
        if origin in config.ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin

    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Request-ID"
    return response


# --- Web UI Routes ---

@app.route("/")
def index():
    """Homepage route showing search statistics."""
    stats = search_service.engine.get_index_statistics()
    doc_count = stats["total_documents"]
    term_count = stats["vocabulary_size"]
    return render_template("index.html", doc_count=doc_count, term_count=term_count)


@app.route("/search")
def web_search():
    """Search route handling queries, pluggable ranking strategies, and pagination."""
    query = request.args.get("q", "").strip()
    ranking = request.args.get("ranking", config.DEFAULT_RANKING_ALGORITHM).strip()
    
    if not query:
        return render_template("results.html", error="Please enter a search term.", query="", ranking=ranking)

    # Execute search with specified ranking strategy
    results = search_service.engine.search(query, log_analytics=True, ranking_algorithm=ranking)
    
    # Handle parsing or ranking errors
    if isinstance(results, dict) and "error" in results:
        return render_template("results.html", error=results["error"], query=query, ranking=ranking)
    
    search_time = round(results.timings.get("total_search_duration", 0.0), 4)
    
    return render_template(
        "results.html", 
        query=query, 
        results=results, 
        search_time=search_time,
        ranking=ranking
    )


@app.route("/analytics")
def analytics_dashboard():
    """Search analytics, system performance, and search quality monitoring dashboard."""
    summary = get_summary_metrics()
    top_queries = get_top_queries(limit=10)
    zero_queries = get_top_zero_result_queries(limit=10)
    type_dist = get_query_type_distribution()
    recent = get_recent_searches(limit=15)
    index_stats = search_service.engine.get_index_statistics()
    memory_stats = get_memory_usage()
    query_cache_stats = search_service.engine.query_cache.get_stats()
    fuzzy_cache_stats = search_service.engine.fuzzy_cache.get_stats()

    # Search Quality Metrics
    quality_report = evaluator.evaluate_engine(search_service.engine, top_k=10, ranking_algorithm="bm25")
    ranking_comparison = evaluator.compare_ranking_methods(search_service.engine)
    fuzzy_tradeoff = evaluator.evaluate_fuzzy_tradeoff(search_service.engine)

    return render_template(
        "analytics.html",
        summary=summary,
        top_queries=top_queries,
        zero_queries=zero_queries,
        type_dist=type_dist,
        recent=recent,
        index_stats=index_stats,
        memory=memory_stats,
        query_cache_stats=query_cache_stats,
        fuzzy_cache_stats=fuzzy_cache_stats,
        quality=quality_report,
        ranking_comparison=ranking_comparison,
        fuzzy_tradeoff=fuzzy_tradeoff,
        bm25_params={"k1": config.BM25_K1, "b": config.BM25_B}
    )


# --- Production Health, Readiness & Metrics Endpoints ---

@app.route("/health")
def health():
    """Liveness & health probe returning search engine health."""
    return jsonify(search_service.engine.health_check())


@app.route("/ready")
def readiness():
    """Readiness probe validating whether index and required models are loaded."""
    health_info = index_manager.get_health()
    status_code = 200 if health_info["ready"] else 503
    return jsonify(health_info), status_code


@app.route("/metrics")
def prometheus_metrics():
    """Prometheus-compatible plain text metrics endpoint."""
    return Response(metrics_registry.to_prometheus(), mimetype="text/plain")


# --- Versioned REST API v1 (`/api/v1/...`) ---

@app.route("/api/v1/search")
def api_v1_search():
    """
    Production Versioned Search API.
    Parameters:
      - q: query text (required)
      - method: ranking algorithm (bm25, tfidf, frequency, ltr, semantic, hybrid, bm25_ltr, hybrid_ltr)
      - top_k: maximum candidate matches (1..100)
      - page: result page index (>= 1)
      - limit: items per page (1..100)
      - alpha: hybrid weight (0.0..1.0)
    """
    query = request.args.get("q", "").strip()
    method = request.args.get("method", config.DEFAULT_RANKING_ALGORITHM).strip()
    top_k = request.args.get("top_k", config.DEFAULT_TOP_K, type=int)
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", config.DEFAULT_PAGE_SIZE, type=int)
    alpha = request.args.get("alpha", type=float)

    response = search_service.search(
        query=query,
        method=method,
        top_k=top_k,
        page=page,
        limit=limit,
        alpha=alpha,
        request_id=request.request_id
    )

    status_code = response.get("status_code", 200)
    return jsonify(response), status_code


@app.route("/api/v1/health")
def api_v1_health():
    return health()


@app.route("/api/v1/ready")
def api_v1_ready():
    return readiness()


@app.route("/api/v1/metrics")
def api_v1_metrics():
    """JSON snapshot of production system metrics."""
    return jsonify(metrics_registry.to_dict())


@app.route("/api/v1/explain")
def api_v1_explain():
    """Score attribution and breakdown endpoint."""
    query = request.args.get("q", "").strip()
    doc_id = request.args.get("doc", "").strip()
    ranking = request.args.get("ranking", config.DEFAULT_RANKING_ALGORITHM).strip()
    k1 = request.args.get("k1", type=float)
    b = request.args.get("b", type=float)
    
    if not query or not doc_id:
        return jsonify({"error": "Missing required parameters: 'q' and 'doc'."}), 400

    explanation = search_service.engine.explain_score(
        query=query, 
        filename=doc_id, 
        ranking_algorithm=ranking,
        k1=k1,
        b=b
    )
    explanation["request_id"] = request.request_id
    return jsonify(explanation)


# --- Legacy API Compatibility Endpoints ---

@app.route("/api/search/explain")
def api_legacy_explain():
    return api_v1_explain()


@app.route("/api/analytics/summary")
def api_summary():
    return jsonify(get_summary_metrics())


@app.route("/api/analytics/top-queries")
def api_top_queries():
    limit = request.args.get("limit", 10, type=int)
    return jsonify(get_top_queries(limit=limit))


@app.route("/api/analytics/performance")
def api_performance():
    summary = get_summary_metrics()
    memory = get_memory_usage()
    return jsonify({
        "latency_percentiles_ms": {
            "avg": summary["avg_latency_ms"],
            "median_p50": summary["median_latency_ms"],
            "p95": summary["p95_latency_ms"],
            "p99": summary["p99_latency_ms"],
            "min": summary["min_latency_ms"],
            "max": summary["max_latency_ms"]
        },
        "memory_allocation": memory,
        "query_cache": search_service.engine.query_cache.get_stats(),
        "fuzzy_cache": search_service.engine.fuzzy_cache.get_stats()
    })


@app.route("/api/analytics/quality")
def api_quality():
    ranking = request.args.get("ranking", config.DEFAULT_RANKING_ALGORITHM)
    report = evaluator.evaluate_engine(search_service.engine, top_k=10, ranking_algorithm=ranking)
    return jsonify(report)


@app.route("/api/analytics/index")
def api_index():
    return jsonify(search_service.engine.get_index_statistics())


@app.route("/api/analytics/cache")
def api_cache():
    return jsonify({
        "query_cache": search_service.engine.query_cache.get_stats(),
        "fuzzy_cache": search_service.engine.fuzzy_cache.get_stats()
    })


@app.route("/api/ltr/status")
def api_ltr_status():
    import json
    if config.LTR_METADATA_PATH.exists():
        try:
            with open(config.LTR_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return jsonify(metadata)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "not_trained", "feature_version": config.FEATURE_VERSION})


@app.route("/api/vector/status")
def api_vector_status():
    import json
    if config.VECTOR_METADATA_PATH.exists():
        try:
            with open(config.VECTOR_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return jsonify(metadata)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "not_built", "embedding_model": config.EMBEDDING_MODEL_NAME})


# --- Central Error Handlers ---

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "request_id": getattr(request, "request_id", uuid.uuid4().hex[:8]),
            "error": "Resource not found.",
            "status_code": 404
        }), 404
    return render_template("404.html"), 404


@app.errorhandler(500)
def handle_500(e):
    req_id = getattr(request, "request_id", uuid.uuid4().hex[:8])
    logger.error(f"[{req_id}] Internal Server Error: {e}", exc_info=False)
    if request.path.startswith("/api/"):
        return jsonify({
            "request_id": req_id,
            "error": "Internal Server Error.",
            "status_code": 500
        }), 500
    return render_template("results.html", error="Internal Server Error occurred. Please try again later.", query=""), 500


if __name__ == "__main__":
    logger.info(f"Starting Mini Search Engine Server in [{config.APP_ENV}] mode on {config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
