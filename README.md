# Mini Search Engine

A high-performance command-line and web-based search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns ranked matching filenames.

Built as a learning project to understand Python fundamentals, Information Retrieval, Algorithms & Data Structures, Machine Learning, Neural/Semantic Vector Search, and Web Development.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index with two-pointer intersection
- Positional Index for exact phrase matching (`"exact phrase"`)
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- Boolean Search (`AND`, `OR`, `NOT`, `( )`) with early termination
- Fuzzy Search & Typo Tolerance (Levenshtein Distance via Dynamic Programming with length pre-filtering)
- **6 Pluggable Search & Ranking Strategies:**
  1. **Frequency Ranking (Sparse TF)**
  2. **TF-IDF Ranking (Sparse TF $\times$ Logarithmic IDF)**
  3. **BM25 Ranking (Probabilistic saturation & length normalization)**
  4. **Learning-to-Rank (LTR Pointwise Logistic Regression with L2 Regularization)**
  5. **Semantic Search (Dense Vector Embeddings & Exact Cosine Similarity Retrieval)**
  6. **Hybrid Search (Sparse BM25 + Dense Semantic Fusion with Min-Max Normalization)**
- **Dense Embedding Service (`EmbeddingService`) with LRU Caching & Batch Encoding**
- **Vector Storage & Exact Nearest Neighbor Index (`NumpyVectorStore`)**
- **Hybrid Retrieval Engine (`HybridSearchEngine`) with configurable $\alpha \in [0.0, 1.0]$**
- **Score Attribution & Explanation Engine (`explain_score` / "Why this result?")**
- **Search Quality & Relevance Testing Framework (MAP, MRR, NDCG@5, NDCG@10, P@K, R@K, F1)**
- High-Performance Query Result Cache (`BoundedLRUCache`) with automated cache invalidation
- Incremental Indexing (`add_document`, `remove_document`) without full corpus re-indexing
- Index Integrity Validation & JSON Serialization (`save_index`, `load_index`, `vector_index.json`)
- Search Analytics & Performance Monitoring (SQLite Event Logging & Dashboard)
- System Health Check Endpoint (`/health`)
- Web Interface (Flask & HTML/CSS) with ranking strategy selector and score explanation

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- SQLite3 (Analytics Storage)
- HTML5 & CSS3
- `numpy`, `pathlib`, `string`, `collections`, `math`, `html`, `re`, `time`, `json`, `heapq`, `tracemalloc`, `csv`, `random`, `hashlib` (Python standard library & scientific computing)

---

## System Architecture (Stage 15)

```text
                         USER
                          │
                          ▼
              Web Interface / API / CLI
                          │
                          ▼
            Query Processing & AST Tokenizer
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Traditional LTR     Semantic Search    Hybrid Search
 (BM25, TF-IDF, TF)  (Dense Embeddings)   (Sparse + Dense)
        │                 │                 │
        │                 ▼                 │
        │         NumpyVectorStore          │
        │         (Cosine Sim Top-K)        │
        │                 │                 │
        └────────┬────────┴─────────────────┘
                 │
                 ▼
          Hybrid Search Engine
         (Candidate Union & Min-Max Fusion)
                 │
                 ▼
           Ranked Results
                 │
                 ▼
          Search Analytics

Offline Indexing & Evaluation:
 ├──► Vector Index Builder (build_vector_index.py) -> Encodes docs & saves vector_index.json
 ├──► LTR Training (train_ltr.py) -> Trains Pointwise & Pairwise Rankers
 ├──► Hybrid Alpha Experiment (experiment_hybrid.py) -> Grid search over alpha in [0.0, 1.0]
 └──► Search Quality Evaluator (evaluate.py) -> 6-way MAP, MRR, NDCG@5 Benchmark
```

---

## 6-Way Ranking Comparison Benchmark

Evaluated across the 25 ground-truth relevance queries using `python evaluate.py`:

| Ranking Algorithm | Representation | MAP | MRR | NDCG@5 | Precision@1 | Recall@5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Learning-to-Rank (LTR)** | Learned (10 features) | **0.9987** | **1.0000** | **0.9948** | **0.9200** | **0.9920** |
| **Frequency Ranking** | Sparse Lexical | **0.9971** | **1.0000** | **0.9942** | **0.9200** | **0.9920** |
| **TF-IDF Ranking** | Sparse Lexical | **0.9951** | **1.0000** | **0.9932** | **0.9200** | **0.9920** |
| **BM25 Ranking ($k_1=1.2$)** | Sparse Probabilistic | **0.9884** | **1.0000** | **0.9900** | **0.9200** | **0.9920** |
| **Hybrid Search ($\alpha=0.75$)** | Sparse + Dense | **0.8893** | **0.9200** | **0.8848** | **0.9200** | **0.9573** |
| **Semantic Search (Dense)** | Dense Vectors (64-dim) | **0.7981** | **0.8333** | **0.8204** | **0.7600** | **0.9573** |

---

## Hybrid Search Alpha Spectrum Experiment

Evaluated using `python experiment_hybrid.py`:

$$S_{\text{hybrid}}(D) = \alpha \cdot S_{\text{bm25, norm}}(D) + (1 - \alpha) \cdot S_{\text{semantic, norm}}(D)$$

| Mode | Alpha ($\alpha$) | MAP | MRR | NDCG@5 | Precision@1 | Recall@5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Semantic** | $0.00$ | 0.7981 | 0.8333 | 0.8204 | 0.7600 | 0.9573 |
| **Hybrid ($\alpha=0.25$)** | $0.25$ | 0.8536 | 0.8800 | 0.8590 | 0.8400 | 0.9573 |
| **Hybrid ($\alpha=0.50$)** | $0.50$ | 0.8849 | 0.9200 | 0.8823 | 0.9200 | 0.9573 |
| **Hybrid ($\alpha=0.75$)** | **0.75** | **0.8893** | **0.9200** | **0.8848** | **0.9200** | **0.9573** |
| **Pure BM25 Candidates** | $1.00$ | 0.8735 | 0.9200 | 0.8786 | 0.9200 | 0.9640 |

---

## How to Install & Run

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the Dense Vector Index:**
   ```bash
   python build_vector_index.py
   ```

4. **Train the Learning-to-Rank Model:**
   ```bash
   python train_ltr.py
   ```

5. **Run the Hybrid Alpha Grid Experiment:**
   ```bash
   python experiment_hybrid.py
   ```

6. **Run the 6-Way Quality Evaluation Suite:**
   ```bash
   python evaluate.py
   ```

7. **Run the Web Application:**
   ```bash
   python app.py
   ```
   - Web Search: `http://127.0.0.1:5000`
   - Analytics & Quality Dashboard: `http://127.0.0.1:5000/analytics`
   - Vector Store Status API: `http://127.0.0.1:5000/api/vector/status`
   - Score Explanation API: `http://127.0.0.1:5000/api/search/explain?q=programming&doc=python.txt&ranking=hybrid`

8. **Run all 112 unit and regression tests:**
   ```bash
   python -m unittest discover tests
   ```

---

## Future Roadmap

This is a multi-stage project:

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
15. ✅ **Stage 15** — Neural / Semantic Search & Vector Retrieval (current)
16. Stage 16 — Search Engine Productionization, Advanced Retrieval Architecture, and Deployment
