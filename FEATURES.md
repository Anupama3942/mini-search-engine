# Feature Inventory

This document outlines all features implemented in the Mini Search Engine project.

## Feature List

| Feature | Status | Description |
| --- | --- | --- |
| Inverted Index | Done | Core data structure for text retrieval. |
| TF-IDF | Done | Term frequency-inverse document frequency ranking. |
| BM25 | Done | Probabilistic ranking model. |
| Boolean Search | Done | AND/OR/NOT query capabilities. |
| Phrase Search | Done | Exact multi-term phrase matching. |
| Fuzzy Search | Done | Levenshtein-distance based fuzzy matching. |
| Spell Correction | Done | Did-you-mean suggestions for misspelled queries. |
| Query Expansion | Done | Synonym expansion to improve recall. |
| Query Understanding | Done | Analyzing and restructuring query parameters. |
| Intent Classification | Done | Classifying query intent to tailor results. |
| Learning-to-Rank | Done | Machine learning model for advanced re-ranking. |
| Semantic Search | Done | Vector-based dense retrieval matching. |
| Vector Retrieval | Done | Nearest-neighbor vector search. |
| Hybrid Search | Done | Combining sparse (BM25) and dense (Semantic) scores. |
| Two-stage Retrieval | Done | Pipeline retrieving candidates then re-ranking. |
| Analytics Dashboard | Done | UI to view search metrics and CTR. |
| A/B Testing | Done | Framework for testing changes using deterministic SHA-256 hashing. |
| REST API v1 | Done | Complete API for search and integrations. |
| Docker Deployment | Done | Containerized application environment. |
| Prometheus Monitoring | Done | Exposed metrics format for monitoring. |
| Rate Limiting | Done | Controls API usage limits per IP. |
| Health/Readiness Probes | Done | Endpoints for load balancer/orchestrator health checks. |
| Caching | Done | LRU caching for queries and fuzzy matching. |
| Security Headers | Done | Hardened HTTP response headers. |

## Search Method Comparison

| Method | Concept | Strength | Weakness | Use Case | MAP | Avg Latency |
| --- | --- | --- | --- | --- | --- | --- |
| Frequency | Raw term count | Very fast, simple | Poor quality | Baseline text search | 0.9971 | - |
| TF-IDF | Weighted term count | Good baseline | Struggles with long docs | Standard keyword search | 0.9951 | - |
| BM25 | Probabilistic term weights | Excellent accuracy, fast | No semantic understanding | Production keyword search | 0.9884 | 9.84ms |
| LTR | Feature-based ML | Highly tunable | Requires training data | Re-ranking | 0.9987 | - |
| Semantic | Dense vector similarity | Captures intent and context | Slower, embedding overhead | Concept-based search | 0.8127 | 13.01ms |
| Hybrid | BM25 + Semantic | Best of both worlds | Highest computational cost | Complex production use | 0.8816 | 9.08ms |
| BM25→LTR | BM25 recall + LTR precision | Excellent quality and speed | Added complexity | Optimal production pipeline | 0.9920 | 2.47ms |
