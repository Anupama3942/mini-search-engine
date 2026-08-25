# Mini Search Engine

A production-grade, high-performance search service and web engine built with Python. Supports classical inverted index search, probabilistic BM25 ranking, Learning-to-Rank (LTR), Neural/Semantic dense vector retrieval, and two-stage hybrid retrieval pipelines.

Built as an engineering project to understand Information Retrieval, Algorithmic Data Structures, Machine Learning, Two-Stage Search Architecture, Observability, and Production Web Services.

---

## Key Features

- **6 Pluggable Search & Ranking Strategies:**
  1. **Frequency Ranking (Sparse TF)**
  2. **TF-IDF Ranking (Sparse TF $\times$ IDF)**
  3. **BM25 Ranking (Probabilistic saturation & document length normalization)**
  4. **Learning-to-Rank (LTR Pointwise Logistic Regression with L2 Regularization)**
  5. **Semantic Search (Dense Vector Embeddings & Exact Cosine Similarity Retrieval)**
  6. **Hybrid Search (Sparse BM25 + Dense Semantic Fusion with Min-Max Normalization)**
- **Two-Stage Production Retrieval Pipelines:**
  - **BM25 $\to$ LTR (Candidate Retrieval $\to$ ML Reranking)**
  - **Hybrid $\to$ LTR (Sparse+Dense Fusion $\to$ ML Reranking)**
- **Production REST API v1 (`/api/v1/...`) with Request ID Tracing (`X-Request-ID`)**
- **Observability & Health Probes:**
  - Liveness probe (`GET /health` & `GET /api/v1/health`)
  - Readiness probe (`GET /ready` & `GET /api/v1/ready`)
  - Prometheus plain text metrics exporter (`GET /metrics`)
  - JSON system metrics snapshot (`GET /api/v1/metrics`)
- **Resilience & Security:**
  - Sliding-window IP rate limiter
  - Security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`)
  - Bounded input validation (`MAX_QUERY_LENGTH=500`, `MAX_TOP_K=100`)
  - Graceful degradation with automatic fallback chain down to BM25
  - Atomic multi-index construction (`build_index.py`)
- **Query Processing:** Boolean search (`AND`, `OR`, `NOT`, `( )`), Exact phrase matching (`"exact phrase"`), Levenshtein typo tolerance.
- **High-Performance Caching:** `BoundedLRUCache` with automated invalidation.
- **Docker & VPS Deployment:** Multi-stage `Dockerfile`, `.dockerignore`, and systemd unit file (`search-engine.service`).

---

## Production System Architecture

```text
                               INTERNET / CLIENT
                                      │
                                      ▼
                               Reverse Proxy
                                (TLS / HTTPS)
                                      │
                                      ▼
                              REST API v1 / Web UI
                         (Security Headers & Rate Limiting)
                                      │
                                      ▼
                                SearchService
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
          BM25Retriever       SemanticRetriever       HybridRetriever
           (Sparse TF)        (Dense Embeddings)     (Candidate Union)
                │                     │                     │
                └─────────────────────┼─────────────────────┘
                                      │
                                      ▼
                              Candidate Pool
                                      │
                                      ▼
                               LTR / Ranker
                                      │
                                      ▼
                              Result Processing
                                 (Pagination)
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
               MetricsRegistry                    Analytics
             (Prometheus Metrics)             (SQLite Events)
```

---

## Production Pipeline Benchmark Comparison

Evaluated across the 25-query ground-truth benchmark (`python benchmark_production.py`):

| Retrieval Pipeline | MAP | MRR | NDCG@5 | Precision@5 | Avg Latency | P95 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 (Sparse)** | **0.9884** | **1.0000** | **0.9900** | **0.4080** | 6.76 ms | 13.87 ms |
| **BM25 $\to$ LTR (Two-Stage)** | **0.9671** | **0.9733** | **0.9738** | **0.4080** | **0.76 ms** | **2.07 ms** |
| **Hybrid (Sparse + Dense)** | **0.8816** | **0.9200** | **0.8806** | **0.3840** | 6.58 ms | 7.89 ms |
| **Semantic (Dense 64-dim)** | **0.8347** | **0.8667** | **0.8467** | **0.3840** | 9.87 ms | 13.93 ms |
| **Hybrid $\to$ LTR (Two-Stage)** | **0.6806** | **0.6793** | **0.7212** | **0.3840** | 4.69 ms | 8.64 ms |

---

## REST API v1 Specification

### 1. Execute Search Query
```http
GET /api/v1/search?q={query}&method={method}&top_k={top_k}&page={page}&limit={limit}
```
**Example Request:**
```bash
curl "http://localhost:5000/api/v1/search?q=python+programming&method=bm25_ltr&top_k=10&page=1&limit=5"
```
**Example Response:**
```json
{
  "request_id": "e802b75f",
  "query": "python programming",
  "method": "bm25_ltr",
  "total_results": 4,
  "page": 1,
  "limit": 5,
  "total_pages": 1,
  "search_duration_seconds": 0.00076,
  "results": [
    {
      "filename": "python.txt",
      "title": "Python",
      "score": 0.9987,
      "snippet": "<mark>Python</mark> is a high-level, general-purpose programming language...",
      "ranking_algorithm": "bm25->ltr"
    }
  ]
}
```

### 2. Health & Readiness Probes
- `GET /health` or `GET /api/v1/health` $\to$ Returns `{"status": "healthy"}`
- `GET /ready` or `GET /api/v1/ready` $\to$ Returns `{"ready": true, "status": "ready"}`

### 3. Prometheus Metrics Exporter
- `GET /metrics` $\to$ Returns Prometheus formatted application counters and P95 latencies.

---

## Installation & Running Locally

1. **Clone repository and create virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build Indexes & Train Models:**
   ```bash
   python build_index.py
   python train_ltr.py
   ```

4. **Run Search Server:**
   ```bash
   python app.py
   ```
   - Homepage: `http://localhost:5000`
   - Analytics Dashboard: `http://localhost:5000/analytics`
   - Prometheus Metrics: `http://localhost:5000/metrics`
   - Readiness Probe: `http://localhost:5000/ready`

5. **Run Full Test Suite (122 tests):**
   ```bash
   python -m unittest discover tests
   ```

---

## Docker Deployment

1. **Build the container image:**
   ```bash
   docker build -t mini-search-engine:latest .
   ```

2. **Run container:**
   ```bash
   docker run -d -p 5000:5000 --name search-engine mini-search-engine:latest
   ```

3. **Verify container health:**
   ```bash
   docker ps
   curl http://localhost:5000/health
   ```

---

## Project Structure

```text
python-search-engine-project/
├── app.py                      # Production Flask Web Server & API v1
├── config.py                   # Central environment configuration
├── search.py                   # Inverted index search engine core
├── benchmark_production.py     # Two-stage pipeline benchmarking CLI
├── build_index.py              # Atomic multi-index build tool
├── train_ltr.py                # LTR training pipeline & ablation
├── evaluate.py                 # 6-way search quality evaluation
├── services/                   # Service layer
│   ├── search_service.py       # Central SearchService & two-stage pipelines
│   ├── index_manager.py        # Centralized atomic IndexManager
│   ├── retrieval.py            # Base, BM25, Semantic, Hybrid retrievers
│   └── metrics.py              # Thread-safe MetricsRegistry & Prometheus
├── semantic/                   # Vector retrieval & embeddings
│   ├── embeddings.py           # EmbeddingService & dense models
│   ├── vector_store.py         # NumpyVectorStore & cosine similarity
│   └── hybrid.py               # HybridSearchEngine & Min-Max fusion
├── ranking/                    # Ranking strategies
│   ├── bm25.py                 # BM25 Ranker
│   ├── ltr.py                  # LTR Ranker with BM25 fallback
│   ├── semantic.py             # Semantic Ranker
│   └── hybrid.py               # Hybrid Ranker
├── evaluation/                 # Relevance judgments & IR metrics
├── templates/                  # Jinja2 HTML templates
├── static/                     # CSS stylesheets
├── tests/                      # 122 Unit, integration, & API tests
├── Dockerfile                  # Production container definition
├── search-engine.service       # Linux systemd unit service file
├── PRODUCTION_CHECKLIST.md     # Deployment verification checklist
└── TROUBLESHOOTING.md          # Diagnostics & disaster recovery
```

---

## Roadmap

1. ✅ **Stage 1** — Basic document search
2. ✅ **Stage 2** — Inverted Index 
3. ✅ **Stage 3** — Text Processing
4. ✅ **Stage 4** — Search Ranking
5. ✅ **Stage 5** — TF-IDF Ranking
6. ✅ **Stage 6** — Web Interface
7. ✅ **Stage 7** — Boolean Search 
8. ✅ **Stage 8** — Phrase Search
9. ✅ **Stage 9** — Fuzzy Search & Typo Tolerance
10. ✅ **Stage 10** — Search Analytics & Performance Monitoring
11. ✅ **Stage 11** — Search Engine & Index Optimization
12. ✅ **Stage 12** — Search Quality Evaluation & Relevance Testing
13. ✅ **Stage 13** — Advanced Ranking & BM25
14. ✅ **Stage 14** — Learning-to-Rank (LTR) & Advanced Ranking Experiments
15. ✅ **Stage 15** — Neural / Semantic Search & Vector Retrieval
16. ✅ **Stage 16** — Search Engine Productionization, Advanced Retrieval Architecture & Deployment (current)
17. Stage 17 — Advanced Search Features, Query Understanding, and Search Experience
