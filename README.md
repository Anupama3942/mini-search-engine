# Mini Search Engine

A production-grade, full-stack information retrieval system built from scratch in Python. Implements classical inverted index search, probabilistic BM25 ranking, Learning-to-Rank (LTR), neural/semantic dense vector retrieval, hybrid search fusion, query understanding, A/B experimentation, and two-stage retrieval pipelines.

Built progressively across 21 stages as an engineering portfolio project covering Information Retrieval, Data Structures, Machine Learning, NLP, Production Architecture, and Search Quality Evaluation.

---

## Features

### Search & Ranking
- **6 Pluggable Ranking Strategies:** Frequency, TF-IDF, BM25, LTR, Semantic, Hybrid
- **Two-Stage Pipelines:** BM25 → LTR and Hybrid → LTR (candidate retrieval + ML reranking)
- **Boolean Search:** `AND`, `OR`, `NOT`, nested parentheses
- **Phrase Search:** Exact positional matching with `"quoted phrases"`
- **Fuzzy Search:** Levenshtein edit distance typo tolerance with "Did you mean?" suggestions

### Query Understanding
- Spell correction with frequency-weighted confidence scoring
- Synonym expansion (conservative and aggressive modes)
- Intent classification (keyword, boolean, phrase, informational, navigational)
- Query-adaptive routing to optimal retrieval strategy
- Autocomplete suggestions (`/api/v1/suggest`)

### Analytics & Experimentation
- Search analytics dashboard with CTR, latency percentiles, and query distributions
- A/B experimentation platform with deterministic SHA-256 variant assignment
- Offline evaluation with Welch's t-test and 95% confidence intervals
- Click-through tracking with position attribution

### Production Infrastructure
- REST API v1 with request ID tracing (`X-Request-ID`)
- Health (`/health`) and readiness (`/ready`) probes
- Prometheus metrics exporter (`/metrics`)
- IP-based sliding-window rate limiter (120 req/min)
- Security headers (CORS, CSP, XSS protection)
- Docker deployment with non-root container user
- Systemd service configuration for Linux VPS
- Atomic multi-index construction with zero-downtime rebuilds

### Quality & Testing
- 25 ground-truth evaluation queries with relevance judgments
- Automated quality gate (MAP ≥ 0.70, MRR ≥ 0.75, P@1 ≥ 0.70)
- 138 unit, integration, regression, and API tests across 15 test suites
- LRU query and fuzzy caching with automated invalidation

---

## Search Quality Results

Evaluated across 25 ground-truth queries (`python evaluate.py`):

| Metric | Value |
| :--- | :--- |
| **Mean Average Precision (MAP)** | **0.9884** |
| **Mean Reciprocal Rank (MRR)** | **1.0000** |
| **NDCG@5** | **0.9900** |
| **NDCG@10** | **0.9948** |
| **Precision@1** | **0.9200** |
| **Recall@5** | **0.9920** |
| **Recall@10** | **1.0000** |
| **F1-Score** | **0.9583** |
| **Quality Gate** | **PASS** |

### Production Pipeline Benchmark

| Pipeline | MAP | MRR | NDCG@5 | Avg Latency | P95 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 (Sparse)** | 0.9884 | 1.0000 | 0.9900 | 9.84 ms | 23.83 ms |
| **BM25 → LTR** | **0.9920** | **1.0000** | **0.9915** | **2.47 ms** | **9.15 ms** |
| **Hybrid (Sparse+Dense)** | 0.8816 | 0.9200 | 0.8806 | 9.08 ms | 16.50 ms |
| **Semantic (Dense)** | 0.8127 | 0.8467 | 0.8309 | 13.01 ms | 24.89 ms |
| **Hybrid → LTR** | 0.7305 | 0.7380 | 0.7717 | 6.50 ms | 15.82 ms |

**Best overall pipeline:** BM25 → LTR (MAP=0.9920, 2.47ms average latency)

---

## Architecture

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
                            Query Understanding
                   (Spell Check → Synonyms → Intent → Routing)
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

## Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.12+** | Core language (educational, stdlib-rich) |
| **Flask 3.0.3** | Web framework, REST API, template rendering |
| **SQLite** | Analytics event storage (zero-config) |
| **Pure Python IR** | Inverted index, BM25, TF-IDF (no black boxes) |
| **Pure Python ML** | LTR logistic regression, embeddings (no sklearn/torch) |
| **Jinja2** | Server-rendered HTML templates |
| **Docker** | Containerized deployment |
| **Prometheus** | Observability metrics format |

> **Note:** The only external dependency is `Flask==3.0.3`. Everything else — BM25, TF-IDF, LTR, embeddings, vector search, evaluation metrics — is implemented from scratch using Python's standard library.

---

## Installation & Running

### 1. Clone & Setup
```bash
git clone https://github.com/Anupama3942/mini-search-engine.git
cd mini-search-engine
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Build Indexes & Train Models
```bash
python build_index.py
python train_ltr.py
```

### 3. Start Server
```bash
python app.py
```
- **Homepage:** http://localhost:5000
- **Analytics Dashboard:** http://localhost:5000/analytics
- **Health Probe:** http://localhost:5000/health
- **Prometheus Metrics:** http://localhost:5000/metrics

### 4. Run Tests (138 tests)
```bash
python -m unittest discover tests
```

### 5. Run Quality Evaluation
```bash
python evaluate.py
python benchmark_production.py
python run_experiment.py
```

---

## Docker Deployment

```bash
# Build
docker build -t mini-search-engine:latest .

# Run
docker run -d -p 5000:5000 --name search-engine mini-search-engine:latest

# Verify
curl http://localhost:5000/health
curl "http://localhost:5000/api/v1/search?q=python&method=bm25"
```

---

## REST API v1

### Search
```bash
curl "http://localhost:5000/api/v1/search?q=python+programming&method=bm25&top_k=10"
```

### Autocomplete
```bash
curl "http://localhost:5000/api/v1/suggest?q=pyth"
```

### Health & Readiness
```bash
curl http://localhost:5000/api/v1/health
curl http://localhost:5000/api/v1/ready
```

### A/B Experiments
```bash
curl http://localhost:5000/api/v1/experiments
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete endpoint documentation.

---

## Example Searches

| Query | Type | Results |
| :--- | :--- | :--- |
| `python` | Keyword | Matches python.txt (rank 1) |
| `python AND programming` | Boolean AND | Intersection of both terms |
| `python OR java` | Boolean OR | Union of documents |
| `"web development"` | Phrase | Exact positional adjacency match |
| `pythn` | Fuzzy (typo) | Corrected to "python" automatically |
| `beginner python` | Semantic | Conceptual matching via embeddings |

---

## Project Structure

```text
python-search-engine-project/
├── app.py                      # Production Flask Web Server & API v1
├── config.py                   # Central environment configuration
├── search.py                   # Inverted index search engine core
├── analytics.py                # Search event logging & CTR analytics
├── build_index.py              # Atomic multi-index build tool
├── train_ltr.py                # LTR training pipeline
├── evaluate.py                 # Search quality evaluation CLI
├── benchmark_production.py     # Production pipeline benchmarking
├── run_experiment.py           # Offline A/B experiment runner
├── services/                   # Service layer
│   ├── search_service.py       # Central SearchService & two-stage pipelines
│   ├── index_manager.py        # Atomic IndexManager
│   ├── retrieval.py            # BM25, Semantic, Hybrid retrievers
│   └── metrics.py              # MetricsRegistry & Prometheus exporter
├── query_understanding/        # NLP query processing pipeline
│   ├── pipeline.py             # Orchestrator (spell → synonyms → intent)
│   ├── spelling.py             # Levenshtein spell correction
│   ├── synonyms.py             # Synonym expansion
│   └── intent.py               # Intent classification & routing
├── semantic/                   # Vector retrieval & embeddings
│   ├── embeddings.py           # Dense embedding generation
│   ├── vector_store.py         # Vector store & cosine similarity
│   └── hybrid.py               # Hybrid fusion engine
├── ranking/                    # Pluggable ranking strategies
│   ├── bm25.py, tfidf.py       # Classical rankers
│   ├── ltr.py, semantic.py      # ML & neural rankers
│   └── hybrid.py               # Hybrid fusion ranker
├── ltr/                        # Learning-to-Rank framework
│   ├── features.py             # Feature extraction (8 features)
│   ├── models.py               # Pointwise & pairwise models
│   └── dataset.py              # Training data generation
├── evaluation/                 # IR evaluation framework
│   ├── metrics.py              # Precision, Recall, MAP, MRR, NDCG
│   ├── evaluator.py            # SearchEvaluator & quality gates
│   └── relevance_judgments.json # 25 ground-truth queries
├── experimentation/            # A/B testing framework
│   ├── models.py               # Experiment definition & SHA-256 hashing
│   ├── registry.py             # Experiment registry
│   ├── statistics.py           # Welch's t-test & confidence intervals
│   └── offline_evaluator.py    # Offline A/B evaluation
├── templates/                  # Jinja2 HTML templates
├── static/css/                 # Stylesheets
├── tests/                      # 138 tests across 15 suites
├── documents/                  # Document corpus (6 files)
├── Dockerfile                  # Production container
├── search-engine.service       # Systemd unit file
├── ARCHITECTURE.md             # System architecture documentation
├── FEATURES.md                 # Complete feature inventory
├── API_DOCUMENTATION.md        # REST API documentation
├── SECURITY.md                 # Security audit
├── EXPERIMENTATION.md          # A/B testing documentation
├── QUERY_UNDERSTANDING.md      # Query understanding documentation
└── INTERVIEW_QUESTIONS.md      # Technical interview preparation
```

---

## Documentation

| Document | Description |
| :--- | :--- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, pipelines, data flow |
| [FEATURES.md](FEATURES.md) | Complete feature inventory & search method comparison |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API endpoints, parameters, examples |
| [SECURITY.md](SECURITY.md) | Security audit & configuration |
| [TECH_STACK.md](TECH_STACK.md) | Technology stack with rationale |
| [SEARCH_ENGINE_NOTES.md](SEARCH_ENGINE_NOTES.md) | Learning documentation & IR concepts |
| [EXPERIMENTATION.md](EXPERIMENTATION.md) | A/B testing & statistical analysis |
| [QUERY_UNDERSTANDING.md](QUERY_UNDERSTANDING.md) | Query processing pipeline |
| [INTERVIEW_QUESTIONS.md](INTERVIEW_QUESTIONS.md) | 30+ technical interview Q&A |
| [PORTFOLIO.md](PORTFOLIO.md) | Portfolio project description |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [FUTURE.md](FUTURE.md) | Future improvement roadmap |

---

## Limitations

- Small document corpus (6 files) — designed for educational demonstration
- Single-node architecture — no distributed indexing or sharding
- Pure Python embeddings (64-dim) — not transformer-based
- Limited LTR training data — small ground-truth dataset
- No authentication or authorization on API endpoints
- No CI/CD pipeline — manual testing and deployment
- No real-time indexing — batch rebuild required for new documents

---

## Roadmap

1. ✅ **Stage 1** — Basic Document Search
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
14. ✅ **Stage 14** — Learning-to-Rank (LTR)
15. ✅ **Stage 15** — Neural / Semantic Search & Vector Retrieval
16. ✅ **Stage 16** — Production Architecture & Deployment
17. ✅ **Stage 17** — Query Understanding & Advanced Search
18. ✅ **Stage 20** — Advanced Search Analytics & A/B Testing
19. ✅ **Stage 21** — Final Capstone, Polish & Portfolio Release

---

## License

[MIT License](LICENSE)
