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
- Precomputed $O(1)$ Search Ranking (TF-IDF & IDF Caching)
- **High-Performance Query Result Cache (`BoundedLRUCache`) with automated cache invalidation**
- **Incremental Indexing (`add_document`, `remove_document`) without full corpus re-indexing**
- **Index Integrity Validation & JSON Serialization (`save_index`, `load_index`)**
- **Search Analytics & Performance Monitoring (SQLite Event Logging & Dashboard)**
- **Information Retrieval Search Quality & Relevance Evaluation Suite (MAP, MRR, P@K, R@K, F1)**
- **System Health Check Endpoint (`/health`)**
- Web Interface (Flask & HTML/CSS) with search term highlighting, analytics, and XSS protection

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- SQLite3 (Analytics Storage)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re`, `time`, `json`, `heapq`, `tracemalloc`, `csv` (Python standard library)

---

## System Architecture (Stage 12)

```text
                         USER
                          │
                          ▼
                   Search Interface
                          │
                          ▼
                    Search Engine
                          │
                          ▼
                  Ranked Results
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
       User/Search Logs          Evaluation System
                                       │
                                       ▼
                              Ground Truth Dataset
                                       │
                                       ▼
                              Relevance Judgments
                                       │
                                       ▼
                              Evaluation Metrics
                                       │
                ┌──────────────────────┼─────────────────────┐
                ▼                      ▼                     ▼
             Precision              Recall                 F1
                │                      │                     │
                └──────────────────────┼─────────────────────┘
                                       ▼
                                      MAP
                                       │
                                       ▼
                                      MRR
                                       │
                                       ▼
                              Quality Dashboard
```

---

## Stage 12 — Search Quality Evaluation & Relevance Testing

Stage 12 introduces comprehensive Information Retrieval (IR) evaluation to quantitatively measure **search result relevance and ranking quality**.

### Evaluation Metrics

| Metric | Definition | Purpose |
| :--- | :--- | :--- |
| **Precision** | $\frac{\|\text{Retrieved} \cap \text{Relevant}\|}{\|\text{Retrieved}\|}$ | Measures how many returned documents are relevant. |
| **Recall** | $\frac{\|\text{Retrieved} \cap \text{Relevant}\|}{\|\text{Relevant}\|}$ | Measures how many relevant documents in the corpus were found. |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing precision and recall. |
| **Precision@K (P@K)** | $\frac{\|\text{Retrieved}[:K] \cap \text{Relevant}\|}{K}$ | Precision restricted to the top $K$ ranked results ($P@1, P@3, P@5, P@10$). |
| **Recall@K (R@K)** | $\frac{\|\text{Retrieved}[:K] \cap \text{Relevant}\|}{\|\text{Relevant}\|}$ | Coverage of relevant documents in the top $K$ results. |
| **Average Precision (AP)** | $\frac{\sum_{k=1}^N P@k \times \text{rel}(k)}{\|\text{Relevant}\|}$ | Rewards placing relevant documents at higher ranks. |
| **Mean Average Precision (MAP)** | $\frac{1}{\|Q\|} \sum_{q \in Q} \text{AP}(q)$ | Macro average ranking quality across all queries. |
| **Reciprocal Rank (RR)** | $\frac{1}{\text{rank of 1st relevant document}}$ | Measures how quickly the first relevant document is found. |
| **Mean Reciprocal Rank (MRR)** | $\frac{1}{\|Q\|} \sum_{q \in Q} \text{RR}(q)$ | Macro average first-relevant rank quality across all queries. |

---

## Evaluation Benchmark Results

Evaluated across 25 ground-truth queries using `python evaluate.py`:

```text
====================================================================
       MINI SEARCH ENGINE - SEARCH QUALITY EVALUATION (STAGE 12)
====================================================================

[1] DATASET INTEGRITY: PASSED (Version: 1.0, Queries: 25)

[2] OVERALL RELEVANCE & RANKING METRICS
--------------------------------------------------------------------
  Precision@1  (P@1)  : 0.9200   |  Recall@5   (R@5)  : 0.9920
  Precision@3  (P@3)  : 0.5733   |  Recall@10  (R@10) : 1.0000
  Precision@5  (P@5)  : 0.4080   |  F1-Score          : 0.9583
  Precision@10 (P@10) : 0.2080   |  Evaluation Time   : 0.043 s
  Mean Avg Precision (MAP) : 0.9951
  Mean Recip. Rank   (MRR) : 1.0000
--------------------------------------------------------------------

[3] QUERY TYPE BREAKDOWN
Query Type       | Count  | MAP      | MRR      | P@1      | R@5     
--------------------------------------------------------------------
Normal           | 10     | 1.0000   | 1.0000   | 0.9000   | 1.0000  
Boolean          | 5      | 1.0000   | 1.0000   | 1.0000   | 1.0000  
Phrase           | 5      | 1.0000   | 1.0000   | 1.0000   | 1.0000  
Fuzzy            | 5      | 0.9753   | 1.0000   | 0.8000   | 0.9600  
--------------------------------------------------------------------

[4] RANKING COMPARISON (TF-IDF vs Raw Frequency)
Ranking Method         | MAP      | MRR      | P@1      | R@5     
--------------------------------------------------------------------
TF-IDF Ranking         | 0.9951   | 1.0000   | 0.9200   | 0.9920  
Frequency Ranking      | 0.9204   | 0.9200   | 0.9200   | 0.9920  
--------------------------------------------------------------------

[5] FUZZY SEARCH TRADE-OFF (Typo Queries)
Mode               | MAP      | MRR      | P@5      | R@5     
--------------------------------------------------------------------
Fuzzy ON (Stage 9) | 0.9753   | 1.0000   | 0.4800   | 0.9600  
Fuzzy OFF (Exact)  | 0.0000   | 0.0000   | 0.0000   | 0.0000  
--------------------------------------------------------------------
```

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
   - Quality API: `http://127.0.0.1:5000/api/analytics/quality`
   - Health Status: `http://127.0.0.1:5000/health`

4. **Run the Search Quality Evaluation Suite:**
   ```bash
   python evaluate.py
   ```

5. **Run the Performance & Scalability Benchmark Suite:**
   ```bash
   python benchmark.py
   ```

6. **Run all unit & regression tests:**
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
12. ✅ **Stage 12** — Search Quality Evaluation & Relevance Testing (current)
13. Stage 13 — Advanced Ranking & BM25
