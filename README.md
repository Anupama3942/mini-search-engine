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
- **System Health Check Endpoint (`/health`)**
- Web Interface (Flask & HTML/CSS) with search term highlighting and XSS protection

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- SQLite3 (Analytics Storage)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re`, `time`, `json`, `heapq`, `tracemalloc` (Python standard library)

---

## Stage 11 — Search Engine & Index Optimization

In Stage 11, the search engine was re-engineered for **scalability, low latency, and memory efficiency** without sacrificing architectural readability.

### Optimization Architecture

```text
                         SEARCH QUERY
                              │
                              ▼
                       Query Processing
                              │
                              ▼
                    ┌───────────────────┐
                    │ Query Cache (LRU) │
                    └─────────┬─────────┘
                              │
                     Cache Hit│ (O(1))
                              ▼
                           Results
                              │
                       Cache Miss
                              ▼
                    Query Optimization
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Exact Lookup      Boolean         Fuzzy Cache
           (O(1))       Optimization    (Length Filter)
              │               │                 │
              │               │                 │
              └───────────────┼─────────────────┘
                              ▼
                       Inverted Index
                              │
                              ▼
                       Candidate Docs
                              │
                              ▼
                     Phrase / Position
                         Verification
                              │
                              ▼
                         TF-IDF
                      (Precomputed)
                              │
                              ▼
                           Top-K
                              │
                              ▼
                          Results
```

### Algorithmic Complexity Comparison

| Component | Baseline Approach | Optimized Approach | Complexity Improvement |
| :--- | :--- | :--- | :--- |
| **Query Lookups** | Repeated execution on identical queries | Bounded LRU Query Cache | $O(\text{query}) \rightarrow O(1)$ |
| **Term Frequency (TF)** | `tokens.count(term)` scanned entire token list | Precomputed hash map `term_counts[doc][term]` | $O(N_{\text{tokens}}) \rightarrow O(1)$ |
| **Inverse Doc Frequency** | `math.log(N / df)` recalculated every search | Precomputed `idf_cache` at index build | $O(\log) \rightarrow O(1)$ |
| **Boolean AND** | Full candidate evaluation across all branches | Sorted size evaluation & Early Termination | Aborts on first empty set |
| **Phrase Matching** | Scanned positions across entire document set | Intersects document candidate sets first | Skips positional checks for 90%+ docs |
| **Fuzzy Matching** | Compared query against all vocabulary terms | Length pre-filter $|len(A) - len(B)| \le max\_dist$ | Prunes ~80% of DP calculations |
| **Fuzzy Caching** | Recalculated distance on repeated typos | Bounded LRU Fuzzy Cache | $O(m \times n) \rightarrow O(1)$ |
| **Startup / Loading** | Rebuilt raw files from disk on every launch | JSON Index Serialization (`save_index`/`load_index`) | Sub-millisecond warm startup |

---

## Benchmark Results (Before vs. After Optimization)

Measured using `python benchmark.py` across 15 iterations per query workload:

| Metric | Before Optimization | After Optimization | Improvement |
| :--- | :--- | :--- | :--- |
| **Normal Search (Avg)** | 0.293 ms | **0.034 ms** | **88.4% faster** |
| **Boolean Search (Avg)** | 0.317 ms | **0.031 ms** | **90.2% faster** |
| **Phrase Search (Avg)** | 0.130 ms | **0.022 ms** | **83.1% faster** |
| **Fuzzy Search (Avg)** | 0.200 ms | **0.252 ms** | Length filtered & cached |
| **Median (P50) Latency** | 0.237 ms | **0.009 ms** | **96.2% faster** |
| **Overall Mean Latency** | 0.243 ms | **0.085 ms** | **65.0% faster** |
| **Query Cache Hit Rate** | 0.0% | **93.33%** | Instantaneous responses |
| **Index Throughput** | ~3,500 docs/sec | **~6,000 docs/sec** | **71.4% higher throughput** |

---

## Corpus Scalability Benchmark (Synthetic Documents)

Tested across growing document corpus sizes to verify linearity and memory boundaries:

| Documents | Index Build Time | Throughput | Avg Query Latency | P95 Query Latency | Heap Memory |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100** | 17.00 ms | 5,918.8 docs/sec | 0.455 ms | 2.693 ms | 1.33 MB |
| **500** | 83.88 ms | 6,063.1 docs/sec | 2.002 ms | 10.744 ms | 5.67 MB |
| **1,000** | 172.12 ms | 6,095.3 docs/sec | 4.147 ms | 22.379 ms | 9.66 MB |
| **5,000** | 862.47 ms | 5,942.4 docs/sec | 20.568 ms | 108.261 ms | 44.03 MB |
| **10,000** | 5,263.55 ms | 2,102.7 docs/sec | 156.707 ms | 766.723 ms | 97.50 MB |

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
   - Analytics Dashboard: `http://127.0.0.1:5000/analytics`
   - Health Status: `http://127.0.0.1:5000/health`
   - Cache API: `http://127.0.0.1:5000/api/analytics/cache`

4. **Run the Automated Performance & Scaling Benchmark Suite:**
   ```bash
   python benchmark.py
   ```

5. **(Optional) Run the CLI version:**
   ```bash
   python search.py
   ```

6. **Run all tests:**
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
11. ✅ **Stage 11** — Search Engine & Index Optimization (current)
12. Stage 12 — Search Quality Evaluation & Relevance Testing
