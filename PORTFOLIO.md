# Mini Search Engine - Portfolio Project

**Project Title:** Mini Search Engine  
**Short Description:** A full-stack information retrieval system built from scratch in Python.

## The Problem
Modern search engines rely on complex interactions between traditional text matching and cutting-edge machine learning. Building a system that demonstrates these Information Retrieval (IR) fundamentals—from basic inverted indexing to neural semantic search—while maintaining high performance and production readiness is a significant engineering challenge.

## The Solution
A 21-stage progressive build of a search engine. The project starts with rudimentary indexing and classical IR, advances through Machine Learning (Learning-to-Rank) and vector/semantic search, and culminates in a production-ready, containerized microservice with complete telemetry and A/B testing frameworks.

## Key Features
*   **Inverted Index Data Structure:** Custom-built $O(1)$ lookup indexing framework.
*   **Classical IR Algorithms:** Implementation of Term Frequency (TF), TF-IDF, and the industry-standard BM25.
*   **Machine Learning Ranking:** Pointwise Logistic Regression for Learning-to-Rank (LTR).
*   **Feature Extraction Pipeline:** Dynamic extraction of query-document signals (BM25, TF-IDF, document length, match ratios).
*   **Semantic Vector Search:** Cosine similarity over dense vector embeddings.
*   **Hybrid Retrieval:** Alpha-fusion combining sparse (BM25) and dense (semantic) scores.
*   **Two-Stage Search Pipelines:** BM25 $\rightarrow$ LTR and Hybrid $\rightarrow$ LTR candidate generation and reranking.
*   **Query Understanding Engine:** Edit-distance spell correction, synonym expansion, and intent classification.
*   **Evaluation Framework:** Automated calculation of Precision, Recall, MAP, MRR, and NDCG@5.
*   **A/B Experimentation Platform:** Deterministic SHA-256 user hashing for consistent variant assignment.
*   **Statistical Analysis Tooling:** Welch's t-test and Confidence Interval calculations for experiment results.
*   **RESTful API (v1):** Flask-based web service exposing search functionalities.
*   **Production Telemetry:** Prometheus metrics integration (latency, QPS, error rates).
*   **Resiliency Patterns:** Configurable rate limiting and liveness/readiness health probes.
*   **Docker Containerization:** Fully containerized deployment with multi-stage builds.
*   **Comprehensive Test Suite:** 138+ unit and integration tests across 15 suites.

## Technology Stack
*   **Language:** Python 3
*   **Web Framework:** Flask
*   **Metrics:** Prometheus (prometheus_client)
*   **Deployment:** Docker
*   **Testing:** `unittest` (Python standard library)

## Technical Highlights
*   **BM25 Optimization:** Tuned $k_1$ and $b$ parameters to handle diverse document lengths efficiently.
*   **Two-Stage Architecture:** Achieved sub-10ms latency by using BM25 for rapid candidate generation followed by complex ML reranking, proving that high precision doesn't require sacrificing speed.
*   **Embeddings & Hybrid Fusion:** Bridged the gap between exact-match and intent-match by mathematically fusing lexical BM25 scores with dense cosine similarity.
*   **Statistically Sound A/B Testing:** Implemented a rigorous experimentation module that guarantees deterministic bucketing without persistent database state.

## Architecture Overview
The system is designed as a modular pipeline. When a query enters the REST API, it passes through the Query Understanding module (spell check, synonyms). It is then routed to a Retriever (BM25, Semantic, or Hybrid) which queries the Index (Inverted or Vector). In two-stage setups, the initial candidate list is passed to the Reranker (LTR model), which extracts features and re-sorts the candidates. The final ranked list is serialized and returned to the client, while Prometheus asynchronously logs telemetry data.

## Results
Tested across 6 distinct document corpora and 25 ground-truth evaluation queries, with robust Quality Gates (MAP $\ge 0.70$, P@1 $\ge 0.70$, R@5 $\ge 0.70$, MRR $\ge 0.75$).

*   **BM25:** MAP=0.9884, MRR=1.0000, NDCG@5=0.9900, Avg Latency=9.84ms
*   **Semantic:** MAP=0.8127, MRR=0.8467, NDCG@5=0.8309, Avg Latency=13.01ms
*   **Hybrid:** MAP=0.8816, MRR=0.9200, NDCG@5=0.8806, Avg Latency=9.08ms
*   **BM25 $\rightarrow$ LTR:** MAP=0.9920, MRR=1.0000, NDCG@5=0.9915, Avg Latency=2.47ms
*   **Hybrid $\rightarrow$ LTR:** MAP=0.7305, MRR=0.7380, NDCG@5=0.7717, Avg Latency=6.50ms
*   **Quality Gate Status:** PASS (P@1: 0.92, R@5: 0.992)
*   **Test Coverage:** 138 tests, all passing.

## What I Learned
*   How to build high-performance search infrastructure from the ground up without relying on external databases like Elasticsearch.
*   The profound impact of document length normalization ($b$) and term frequency saturation ($k_1$) on search relevance.
*   The mechanics of translating unstructured text into mathematical representations via TF-IDF and dense embeddings.
*   How to architect multi-stage retrieval pipelines to balance computational cost with ranking accuracy.
*   The statistical rigor required to correctly implement and evaluate A/B tests in a production environment.

## Challenges
*   **Memory Efficiency:** Managing in-memory posting lists for the inverted index without unbounded memory growth.
*   **Scoring Normalization:** Fusing BM25 scores (unbounded) with Cosine Similarity scores (-1 to 1) required careful mathematical scaling during hybrid retrieval.
*   **Feature Engineering:** Designing LTR features that were fast to extract but predictive enough to improve over baseline BM25.
*   **Evaluation Rigor:** Building a ground-truth dataset and evaluation loop that correctly penalized false positives while rewarding high-ranked true positives.
*   **Deterministic Testing:** Ensuring random state and hashes were perfectly seeded so that A/B testing logic could be reliably unit tested.

## Future Improvements
*   Implement pairwise and listwise Learning-to-Rank models (e.g., LambdaMART).
*   Add persistence to the index by writing posting lists to disk (e.g., using RocksDB or a custom binary format).
*   Integrate a real embedding model (like SentenceTransformers) to replace the current mock/placeholder embeddings.
*   Implement HNSW (Hierarchical Navigable Small World) graphs for approximate nearest neighbor (ANN) vector search at scale.
