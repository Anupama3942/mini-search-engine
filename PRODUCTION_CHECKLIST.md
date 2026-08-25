# Production Deployment Checklist

Before deploying the search engine to staging or production, ensure each item below is verified.

## 1. Security & Configuration
- [x] `DEBUG` mode is disabled (`APP_ENV=production`, `DEBUG=false`).
- [x] Secrets and environment variables are externalized into `.env` or container environment.
- [x] `.env` is listed in `.gitignore` and `.dockerignore`.
- [x] CORS allowed origins configured to authorized domains.
- [x] Security headers enabled (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`).
- [x] Sliding-window rate limiter enabled to protect against denial-of-service.
- [x] Input query length bounded (`MAX_QUERY_LENGTH=500`).
- [x] Pagination limits bounded (`MAX_TOP_K=100`, `MIN_TOP_K=1`).

## 2. Indexes & Models
- [x] Corpus text documents verified in `documents/`.
- [x] Multi-index build script executed (`python build_index.py`).
- [x] Vector store precomputed and verified in `models/vector_index.json`.
- [x] LTR model trained and verified in `models/ltr_model.json`.
- [x] Model metadata and feature versions validated (`FEATURE_VERSION=1.0`).

## 3. Observability & Reliability
- [x] Liveness probe endpoint operational (`GET /health` returning HTTP 200).
- [x] Readiness probe operational (`GET /ready` returning HTTP 200).
- [x] Prometheus metrics endpoint operational (`GET /metrics`).
- [x] JSON system metrics operational (`GET /api/v1/metrics`).
- [x] Structured request logging enabled with `X-Request-ID` correlation.
- [x] Graceful degradation: Automatic fallback to BM25 if ML models are unavailable.

## 4. Search Quality & Performance
- [x] Automated test suite passed with 100% success (`python -m unittest discover tests`).
- [x] Search quality evaluation gate passed (`python evaluate.py`).
- [x] Two-stage retrieval benchmark completed (`python benchmark_production.py`).
- [x] P95 latency is within target threshold (< 25ms).

## 5. Deployment & Containerization
- [x] Dockerfile verified with non-root security (`USER appuser`).
- [x] Docker healthcheck configured.
- [x] Systemd service configuration verified (`search-engine.service`).
- [x] Reverse proxy TLS/HTTPS termination documented.
