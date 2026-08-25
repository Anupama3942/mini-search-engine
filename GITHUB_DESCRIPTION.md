# Mini Search Engine

> A high-performance, full-stack information retrieval system and search engine built from scratch in Python.

## Features
*   **Core Retrieval:** Inverted indexes, TF-IDF, and industry-standard BM25.
*   **Neural Search:** Semantic vector search and Alpha-fusion Hybrid retrieval.
*   **Machine Learning:** Learning-to-Rank (LTR) with Pointwise Logistic Regression.
*   **Two-Stage Pipelines:** Fast candidate generation (BM25) followed by ML reranking.
*   **Query Understanding:** Spell correction, synonym expansion, and intent classification.
*   **Experimentation:** Deterministic A/B testing and rigorous IR evaluation metrics (MAP, MRR, NDCG).
*   **Production Ready:** Dockerized, REST API, Prometheus metrics, and health probes.

## Architecture Summary
The engine operates as a modular, containerized Flask microservice. Queries undergo lexical processing and intent classification before being routed to one of six ranking strategies (e.g., Hybrid or BM25). For efficiency, two-stage pipelines retrieve a wide net of candidates which are subsequently reranked using extracted ML features. Telemetry and A/B test assignments are handled deterministically at the edge, ensuring consistent user experiences and reliable performance metrics.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Anupama3942/python-search-engine-project.git
cd python-search-engine-project

# 2. Install dependencies (Flask is the only requirement)
pip install -r requirements.txt

# 3. Run the application
python run.py

# 4. Test the API
curl "http://localhost:5000/api/v1/search?q=python&user_id=123"
```

## Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=Prometheus&logoColor=white)
