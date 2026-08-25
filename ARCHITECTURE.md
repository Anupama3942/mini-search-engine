# System Architecture

This document describes the architectural layout of the Mini Search Engine project.

## System Architecture Diagram

```text
+----------+      HTTP       +---------------+
|  Client  | <-------------> |   Flask App   |
+----------+                 +-------+-------+
                                     |
                                     v
                             +---------------+
                             | Search Service|
                             +-------+-------+
                                     |
                                     v
                        +------------------------+
                        |   Query Understanding  |
                        | (Spell, Synonyms, etc) |
                        +------------------------+
                                     |
                                     v
                     +-------------------------------+
                     |     Candidate Retrieval       |
                     |  (BM25 / Semantic / Hybrid)   |
                     +-------------------------------+
                                     |
                                     v
                             +---------------+
                             | Ranking (LTR) |
                             +-------+-------+
                                     |
                                     v
                             +---------------+
                             |   Analytics   |
                             |   & Storage   |
                             +---------------+
```

## Complete Search Pipeline Flow

1. **User Query:** The client submits a search request.
2. **Validation:** The API layer validates the request parameters.
3. **Query Understanding:** The system analyzes the query, extracting intents, expanding synonyms, and checking spelling.
4. **Processing:** The query is tokenized and processed.
5. **Candidate Retrieval:** Initial candidate documents are retrieved using the specified strategy (BM25, Semantic, or Hybrid).
6. **LTR Ranking:** The candidate pool is optionally re-ranked using Learning-to-Rank algorithms (e.g., BM25→LTR, Hybrid→LTR).
7. **Result Processing:** Results are formatted, and metadata/snippets are attached.
8. **Analytics:** The query and basic interactions are logged for experimentation and metrics.
9. **Response:** Formatted results are returned to the client.

## Complete Indexing Pipeline Flow

### Text/Keyword Indexing
- **Documents → Text Processing:** Text normalization, stopword removal.
- **Tokenization:** Breaking text into indexed terms.
- **Inverted Index + BM25 Stats:** Creating the inverted index and pre-computing BM25 statistics.

### Semantic Indexing
- **Documents → Embedding Model:** Documents are processed to generate fixed-size vector embeddings.
- **Vector Index:** Embeddings are stored and mapped against document IDs.

### LTR Training
- **LTR Model → Ranking Features:** The system trains weights to map query-document interaction features to relevance scores.

## Component Responsibilities

| Component | Responsibility |
| --- | --- |
| Flask App | Handles HTTP requests, input validation, and routing. |
| Search Service | Orchestrates the search pipeline and interactions between sub-modules. |
| Query Understanding | Intent classification, spell correction, and query expansion. |
| Candidate Retrieval | Core retrieval logic using various inverted and vector indices. |
| Ranking (LTR) | Re-ranking initial candidate pools to improve precision. |
| Analytics | Logging interactions and managing A/B experiments. |
| Storage/Indices | In-memory and SQLite-backed storage of documents and metrics. |

## Module Dependency Map
- `app.py` depends on `api/`, `services/`, and `config.py`.
- `services/search.py` depends on `retrieval/`, `ranking/`, and `query_understanding/`.
- `retrieval/` depends on `index/` (inverted, vector).
- `analytics/` depends on `db/` (SQLite interactions).
