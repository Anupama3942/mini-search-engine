# Resume Descriptions

## Option 1: One-Line Bullet (For dense skill-based resumes)
* Engineered a Python search engine microservice featuring BM25, semantic hybrid retrieval, and Learning-to-Rank, achieving MAP=0.99 and sub-10ms latency via Docker and Flask.

## Option 2: Three-Bullet Description (For standard project sections)
* **Mini Search Engine**: Built a production-ready information retrieval system from scratch in Python, implementing custom inverted indexes, BM25, and semantic vector search.
* Designed two-stage retrieval pipelines (BM25 $\rightarrow$ LTR) utilizing Pointwise Logistic Regression for reranking, combined with spell correction and A/B testing frameworks.
* Deployed as a Dockerized Flask REST API with Prometheus metrics; rigorously validated via 138 unit tests, achieving a Mean Average Precision (MAP) of 0.9920 and sub-10ms average latency.

## Option 3: Detailed Description (For specialized portfolios/roles)
**Mini Search Engine (Python, Flask, Docker, ML)**
Architected and implemented a full-stack search engine from the ground up to demonstrate advanced Information Retrieval concepts. Developed a highly optimized inverted index supporting TF-IDF and BM25 scoring for exact keyword matching, and integrated dense embeddings for semantic search capabilities. Combined these approaches using Alpha-fusion hybrid retrieval and engineered two-stage pipelines where fast BM25 candidate generation is followed by a Machine Learning (LTR) reranking phase. Built comprehensive query understanding modules including spell correction and intent classification. The system was validated against a ground-truth dataset across 6 document corpora, yielding a MAP of 0.9920 and 100% test pass rate across 138 suites. Containerized with Docker, the service exposes a REST API fortified with Prometheus metrics, rate limiting, and robust A/B testing infrastructure.
