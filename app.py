"""
Mini Search Engine - Stage 13 (Web Application)
Flask Web Server with Search Analytics, Performance Dashboard, Health Checks, Quality Evaluation, BM25 Explain, and API Endpoints.
"""

from flask import Flask, render_template, request, jsonify
import time
import config
from search import SearchEngine
from analytics import (
    get_summary_metrics,
    get_top_queries,
    get_top_zero_result_queries,
    get_query_type_distribution,
    get_recent_searches
)
from performance import get_memory_usage
from evaluation.evaluator import SearchEvaluator

app = Flask(__name__)

# Initialize search engine and evaluator in memory
search_engine = SearchEngine()
evaluator = SearchEvaluator()


@app.route("/")
def index():
    """Homepage route showing search statistics."""
    stats = search_engine.get_index_statistics()
    doc_count = stats["total_documents"]
    term_count = stats["vocabulary_size"]
    return render_template("index.html", doc_count=doc_count, term_count=term_count)


@app.route("/search")
def search():
    """Search route handling queries, pluggable ranking strategies, caching, and analytics."""
    query = request.args.get("q", "").strip()
    ranking = request.args.get("ranking", config.DEFAULT_RANKING_ALGORITHM).strip()
    
    if not query:
        return render_template("results.html", error="Please enter a search term.", query="", ranking=ranking)

    # Execute search with specified ranking strategy
    results = search_engine.search(query, log_analytics=True, ranking_algorithm=ranking)
    
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
    index_stats = search_engine.get_index_statistics()
    memory_stats = get_memory_usage()
    query_cache_stats = search_engine.query_cache.get_stats()
    fuzzy_cache_stats = search_engine.fuzzy_cache.get_stats()

    # Search Quality Metrics (Stage 12 & 13)
    quality_report = evaluator.evaluate_engine(search_engine, top_k=10, ranking_algorithm="bm25")
    ranking_comparison = evaluator.compare_ranking_methods(search_engine)
    fuzzy_tradeoff = evaluator.evaluate_fuzzy_tradeoff(search_engine)

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


@app.route("/health")
def health():
    """System health, index consistency, and status endpoint."""
    return jsonify(search_engine.health_check())


# --- REST API Endpoints ---

@app.route("/api/search/explain")
def api_explain_score():
    """JSON API endpoint explaining how a document's ranking score was calculated."""
    query = request.args.get("q", "").strip()
    doc_id = request.args.get("doc", "").strip()
    ranking = request.args.get("ranking", config.DEFAULT_RANKING_ALGORITHM).strip()
    k1 = request.args.get("k1", type=float)
    b = request.args.get("b", type=float)
    
    if not query or not doc_id:
        return jsonify({"error": "Missing required parameters: 'q' and 'doc'."}), 400

    explanation = search_engine.explain_score(
        query=query, 
        filename=doc_id, 
        ranking_algorithm=ranking,
        k1=k1,
        b=b
    )
    return jsonify(explanation)


@app.route("/api/analytics/summary")
def api_summary():
    """JSON API endpoint returning high-level search metrics."""
    return jsonify(get_summary_metrics())


@app.route("/api/analytics/top-queries")
def api_top_queries():
    """JSON API endpoint returning popular search queries."""
    limit = request.args.get("limit", 10, type=int)
    return jsonify(get_top_queries(limit=limit))


@app.route("/api/analytics/performance")
def api_performance():
    """JSON API endpoint returning search latency percentiles and memory usage."""
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
        "query_cache": search_engine.query_cache.get_stats(),
        "fuzzy_cache": search_engine.fuzzy_cache.get_stats()
    })


@app.route("/api/analytics/quality")
def api_quality():
    """JSON API endpoint returning Information Retrieval relevance and quality metrics."""
    ranking = request.args.get("ranking", config.DEFAULT_RANKING_ALGORITHM)
    report = evaluator.evaluate_engine(search_engine, top_k=10, ranking_algorithm=ranking)
    return jsonify(report)


@app.route("/api/analytics/index")
def api_index():
    """JSON API endpoint returning index structural statistics."""
    return jsonify(search_engine.get_index_statistics())


@app.route("/api/analytics/cache")
def api_cache():
    """JSON API endpoint returning cache hit-rate metrics."""
    return jsonify({
        "query_cache": search_engine.query_cache.get_stats(),
        "fuzzy_cache": search_engine.fuzzy_cache.get_stats()
    })


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
