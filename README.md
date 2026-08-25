# Mini Search Engine

A high-performance command-line and web-based search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns ranked matching filenames.

Built as a learning project to understand Python fundamentals, Information Retrieval, Algorithms & Data Structures, Machine Learning, and Web Development.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index with two-pointer intersection
- Positional Index for exact phrase matching (`"exact phrase"`)
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- Boolean Search (`AND`, `OR`, `NOT`, `( )`) with early termination
- Fuzzy Search & Typo Tolerance (Levenshtein Distance via Dynamic Programming with length pre-filtering)
- **Pluggable Multi-Strategy Ranking: Learning-to-Rank (LTR), BM25, TF-IDF, and Term Frequency**
- **Machine-Learned Ranking Model (Pointwise Logistic Regression with L2 Regularization & Feature Normalization)**
- **10 Ranking Features: BM25, TF-IDF, Query Coverage, Exact Matches, TF Sum, Doc Length, Query Length, Phrase Match, Title Match, Fuzzy Similarity**
- **Query-Grouped Training Dataset with Zero-Leakage Train/Val/Test Splits (`train_ltr.py`)**
- **Pairwise Ranker & Feature Ablation Experimentation Suite**
- **Score Attribution & Explanation Engine (`explain_score` / "Why this result?")**
- **Information Retrieval Search Quality & Ranking Evaluation Suite (MAP, MRR, NDCG@5, NDCG@10, P@K, R@K, F1)**
- High-Performance Query Result Cache (`BoundedLRUCache`) with automated cache invalidation
- Incremental Indexing (`add_document`, `remove_document`) without full corpus re-indexing
- Index Integrity Validation & JSON Serialization (`save_index`, `load_index`)
- Search Analytics & Performance Monitoring (SQLite Event Logging & Dashboard)
- System Health Check Endpoint (`/health`)
- Web Interface (Flask & HTML/CSS) with ranking strategy selector and score explanation

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- SQLite3 (Analytics Storage)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re`, `time`, `json`, `heapq`, `tracemalloc`, `csv`, `random` (Python standard library)

---

## System Architecture (Stage 14)

```text
                         USER
                          │
                          ▼
              Web Interface / API / CLI
                          │
                          ▼
            Query Processing & AST Tokenizer
                          │
                          ▼
          Boolean / Phrase / Fuzzy Resolvers
                          │
                          ▼
          Inverted Index & Candidate Retrieval
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    Traditional Rankers           LTR Ranker
  ├── Frequency                1. Feature Extraction (10 features)
  ├── TF-IDF                   2. Feature Normalization (MinMax)
  └── BM25                     3. ML Model (Logistic Regression)
                               4. Predicted Relevance Probability
                               5. Graceful BM25 Fallback
                                        │
                                        ▼
                                  Ranked Results
                                        │
                                        ▼
                                 Search Analytics

Evaluation & Experimentation:
 ├──► Quality Evaluator (evaluate.py) -> MAP, MRR, NDCG@5, P@1, R@5
 ├──► LTR Training & Ablation (train_ltr.py) -> Grid Search, Feature Weights
 └──► BM25 Parameter Tuner (tune.py) -> k1 & b Optimization
```

---

## Stage 14 — Learning-to-Rank (LTR) & Advanced Ranking Experiments

Stage 14 transitions the search engine from manually tuned scoring formulas to **Learning-to-Rank (LTR)**, combining multiple relevance signals into a learned probabilistic model.

### 1. The 10 Ranking Features (`FEATURE_VERSION = "1.0"`)

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `bm25_score` | Continuous | Probabilistic BM25 score ($k_1=1.2, b=0.75$). |
| `tfidf_score` | Continuous | Normalized TF $\times$ logarithmic IDF score. |
| `query_term_coverage` | Ratio ($0.0 - 1.0$) | Fraction of query terms present in the document. |
| `exact_term_match_count`| Integer | Count of distinct query terms appearing in the document. |
| `term_frequency_sum` | Integer | Total frequency count of query terms in the document. |
| `document_length_norm` | Continuous | Document token count relative to average document length ($\frac{\|D\|}{\text{avgdl}}$). |
| `query_length` | Integer | Total number of query tokens. |
| `phrase_match` | Binary ($0/1$) | Indicator whether full query appears as a consecutive phrase. |
| `title_match` | Binary ($0/1$) | Indicator whether query terms match document title/filename. |
| `fuzzy_score` | Ratio ($0.0 - 1.0$) | Average Levenshtein similarity for fuzzy corrected terms. |

---

## 4-Way Ranking Comparison Benchmark

Evaluated across 25 ground-truth queries using `python evaluate.py`:

| Ranking Algorithm | MAP | MRR | NDCG@5 | Precision@1 | Recall@5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Learning-to-Rank (LTR)** | **0.9987** | **1.0000** | **0.9948** | **0.9200** | **0.9920** |
| **Frequency Ranking** | **0.9971** | **1.0000** | **0.9942** | **0.9200** | **0.9920** |
| **TF-IDF Ranking** | **0.9951** | **1.0000** | **0.9932** | **0.9200** | **0.9920** |
| **BM25 Ranking ($k_1=1.2$)** | **0.9884** | **1.0000** | **0.9900** | **0.9200** | **0.9920** |

### Feature Ablation Experiments (`train_ltr.py`)

Evaluating the marginal impact of adding ranking features:

| Feature Set | Feature Count | Test MAP | Test MRR | Test NDCG@5 | Test P@5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 only** | 1 | 1.0000 | 1.0000 | 1.0000 | 0.3600 |
| **BM25 + TF-IDF** | 2 | 1.0000 | 1.0000 | 1.0000 | 0.3600 |
| **+ Coverage** | 3 | 1.0000 | 1.0000 | 1.0000 | 0.3600 |
| **All Features** | 10 | 1.0000 | 1.0000 | 1.0000 | 0.3600 |

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

3. **Train the Learning-to-Rank Model:**
   ```bash
   python train_ltr.py
   ```

4. **Run the 4-Way Quality Evaluation Suite:**
   ```bash
   python evaluate.py
   ```

5. **Run the Web Application:**
   ```bash
   python app.py
   ```
   - Search Homepage: `http://127.0.0.1:5000`
   - Analytics & Quality Dashboard: `http://127.0.0.1:5000/analytics`
   - LTR Model Status API: `http://127.0.0.1:5000/api/ltr/status`
   - Score Explanation API: `http://127.0.0.1:5000/api/search/explain?q=python&doc=python.txt&ranking=ltr`

6. **Run all 103 unit and regression tests:**
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
14. ✅ **Stage 14** — Learning-to-Rank (LTR) & Advanced Ranking Experiments (current)
15. Stage 15 — Neural/Semantic Search and Vector Retrieval
