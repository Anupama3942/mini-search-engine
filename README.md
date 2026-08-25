# Mini Search Engine

A high-performance command-line and web-based search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns ranked matching filenames.

Built as a learning project to understand Python fundamentals, Information Retrieval, Algorithms & Data Structures, and Web Development.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index with two-pointer intersection
- Positional Index for exact phrase matching (`"exact phrase"`)
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- Boolean Search (`AND`, `OR`, `NOT`, `( )`) with early termination
- Fuzzy Search & Typo Tolerance (Levenshtein Distance via Dynamic Programming with length pre-filtering)
- **Pluggable Ranking Architecture: BM25 (Default), TF-IDF, and Term Frequency**
- **Probabilistic BM25 Relevance Scoring with Term Frequency Saturation ($k_1$) and Length Normalization ($b$)**
- **Score Attribution & Explanation Engine (`explain_score` / "Why this result?")**
- **BM25 Parameter Grid Search & Tuner (`tune.py`)**
- High-Performance Query Result Cache (`BoundedLRUCache`) with automated cache invalidation
- Incremental Indexing (`add_document`, `remove_document`) without full corpus re-indexing
- Index Integrity Validation & JSON Serialization (`save_index`, `load_index`)
- Search Analytics & Performance Monitoring (SQLite Event Logging & Dashboard)
- Information Retrieval Search Quality & Relevance Evaluation Suite (MAP, MRR, P@K, R@K, F1)
- System Health Check Endpoint (`/health`)
- Web Interface (Flask & HTML/CSS) with ranking algorithm selector and score breakdown

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- SQLite3 (Analytics Storage)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re`, `time`, `json`, `heapq`, `tracemalloc`, `csv` (Python standard library)

---

## System Architecture (Stage 13)

```text
User
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
 ▼
Ranking Strategy Pattern
 ├── BM25Ranker (Default, k1=1.2, b=0.75)
 ├── TFIDFRanker (Classical TF * log-IDF)
 └── FrequencyRanker (Term Frequency Baseline)
 │
 ▼
Ranked Results & Snippet Generator
 │
 ├──► Score Explanation (explain_score)
 └──► Search Analytics Event Log

Evaluation & Parameter Tuning:
 ├──► Quality Evaluator (evaluate.py) -> MAP, MRR, P@1, R@5
 └──► BM25 Parameter Tuner (tune.py) -> Grid Search over k1 & b
```

---

## Stage 13 — Advanced Ranking & BM25

Stage 13 upgrades the ranking engine from linear TF-IDF to **Best Matching 25 (BM25)**, the standard probabilistic retrieval model in modern search engines.

### 1. Why BM25?
Classical TF-IDF suffers from two key limitations:
1. **Unbounded Term Frequency**: In pure TF-IDF, a document mentioning a term 20 times scores roughly 20 times higher than one mentioning it once. In reality, after 2–3 mentions, additional occurrences provide diminishing relevance.
2. **Lack of Proper Length Normalization**: Long, unfocused documents with hundreds of words can artificially accumulate high term counts solely due to their volume.

BM25 solves both by introducing **Term Frequency Saturation ($k_1$)** and **Document Length Normalization ($b$)**.

### 2. BM25 Scoring Formula

For a query $Q = \{q_1, q_2, \dots\}$ and document $D$:

$$\text{BM25}(D, Q) = \sum_{q_i \in Q} \text{IDF}_{\text{BM25}}(q_i) \times \frac{\text{TF}(q_i, D) \times (k_1 + 1)}{\text{TF}(q_i, D) + k_1 \times \left(1 - b + b \times \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $\text{TF}(q_i, D)$: Raw integer count of query term $q_i$ in document $D$.
- $|D|$: Document length in tokens.
- $\text{avgdl}$: Average document length across the entire corpus ($\frac{\sum |D|}{N}$).
- $k_1$: Term frequency saturation parameter (default: `1.2`).
- $b$: Document length normalization parameter (default: `0.75`).
- $\text{IDF}_{\text{BM25}}(q_i)$: Non-negative Robertson-Spärck Jones Inverse Document Frequency:
  $$\text{IDF}_{\text{BM25}}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$$

---

## Ranking Comparison Benchmark

Evaluated across 25 ground-truth queries using `python evaluate.py`:

| Ranking Algorithm | MAP | MRR | Precision@1 | Recall@5 |
| :--- | :--- | :--- | :--- | :--- |
| **BM25 Ranking ($k_1=1.2, b=0.75$)** | **0.9884** | **1.0000** | **0.9200** | **0.9920** |
| **TF-IDF Ranking** | **0.9951** | **1.0000** | **0.9200** | **0.9920** |
| **Term Frequency Ranking** | **0.9971** | **1.0000** | **0.9200** | **0.9920** |

### BM25 Hyperparameter Tuning Grid (`tune.py`)

Grid search across 25 parameter configurations:

| $k_1$ | $b$ | MAP | MRR | Precision@1 | Recall@5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2.00** | **0.00** | **0.9971** | **1.0000** | **0.9200** | **0.9920** |
| **1.50** | **1.00** | **0.9951** | **1.0000** | **0.9200** | **0.9920** |
| **1.20** | **0.75** (Default) | **0.9884** | **1.0000** | **0.9200** | **0.9920** |
| **0.80** | **0.75** | **0.9884** | **1.0000** | **0.9200** | **0.9920** |

---

## How to Install & Run

1. **Create and activate a virtual environment (Recommended):**
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

3. **Run the Web Application:**
   ```bash
   python app.py
   ```
   - Search Homepage: `http://127.0.0.1:5000`
   - Analytics & Quality Dashboard: `http://127.0.0.1:5000/analytics`
   - Score Explanation API: `http://127.0.0.1:5000/api/search/explain?q=python&doc=python.txt&ranking=bm25`
   - Quality API: `http://127.0.0.1:5000/api/analytics/quality`
   - Health Status: `http://127.0.0.1:5000/health`

4. **Run the BM25 Parameter Tuner:**
   ```bash
   python tune.py
   ```

5. **Run the Search Quality Evaluation Suite:**
   ```bash
   python evaluate.py
   ```

6. **Run the Performance Benchmark Suite:**
   ```bash
   python benchmark.py
   ```

7. **Run all 92 unit and regression tests:**
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
13. ✅ **Stage 13** — Advanced Ranking & BM25 (current)
14. Stage 14 — Learning-to-Rank & Advanced Ranking Experiments
