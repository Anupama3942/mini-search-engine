# Mini Search Engine

A command-line and web-based search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals, Information Retrieval, and Web Development.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index
- Positional Index for exact phrase matching (`"exact phrase"`)
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- Boolean Search (`AND`, `OR`, `NOT`, `( )`)
- Fuzzy Search & Typo Tolerance (Levenshtein Distance via Dynamic Programming)
- Search Ranking (TF-IDF)
- Web Interface (Flask & HTML/CSS)
- **Search Analytics & Performance Monitoring (SQLite Event Logging & Dashboard)**
- **Automated Latency Benchmarks & Percentile Profiling (P50, P95, P99)**
- **System Memory Allocation Tracking (`tracemalloc`)**
- Snippet generation and query term highlighting
- Graceful error handling and XSS protection

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- SQLite3 (Analytics Storage)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re`, `time`, `tracemalloc` (Python standard library)

---

## Stage 10 — Search Analytics & Performance Monitoring

In Stage 10, an observability and performance monitoring layer was introduced to measure query patterns, index construction efficiency, query latencies, and memory consumption.

### Architecture

```text
                         User
                          │
                          ▼
                    Search Interface
                          │
                          ▼
                    Search Engine
                          │
               ┌──────────┴──────────┐
               │                     │
               ▼                     ▼
           Results              Performance
               │                 Measurement
               │                     │
               └──────────┬──────────┘
                          ▼
                   Search Event
                          │
                          ▼
                  Analytics Storage (SQLite)
                          │
                          ▼
                  Analytics Engine
                          │
                          ▼
                 Analytics Dashboard (/analytics)
```

### Metrics Recorded & Monitored

| Metric | Meaning |
| :--- | :--- |
| **Total Searches** | Total number of executed search queries |
| **Average Latency** | Mean search execution time (measured via `time.perf_counter()`) |
| **Median (P50) Latency** | Middle latency where 50% of searches finish at or below this duration |
| **95th Percentile (P95)** | 95% of searches finish at or below this duration (critical for tail latency analysis) |
| **Zero Result Rate** | Percentage of queries that returned 0 matching documents |
| **Fuzzy Usage Rate** | Percentage of queries that triggered Levenshtein typo corrections |
| **Phrase Usage Rate** | Percentage of queries utilizing positional phrase syntax (`"..."`) |
| **Boolean Usage Rate** | Percentage of queries utilizing logical operators (`AND`, `OR`, `NOT`) |
| **Vocabulary Size** | Total count of unique terms indexed across all documents |
| **Total Documents** | Number of `.txt` documents loaded into the corpus |
| **Index Build Time** | Time required to process tokens and build inverted & positional indexes |
| **Indexing Throughput** | Speed of indexing expressed in `documents / second` |
| **Memory Allocation** | Active and peak Python heap memory monitored via `tracemalloc` |

### Search Event JSON Example

Every search generates an anonymous performance event:

```json
{
  "timestamp": "2026-08-25T08:12:30.123456Z",
  "query": "pythn AND programing",
  "normalized_query": "pythn and programing",
  "result_count": 2,
  "search_duration": 0.000656,
  "query_parsing_time": 0.000045,
  "term_resolution_time": 0.000382,
  "retrieval_time": 0.000115,
  "ranking_time": 0.000114,
  "query_type": "boolean + fuzzy",
  "fuzzy_used": true,
  "phrase_used": false,
  "boolean_used": true,
  "zero_result": false
}
```

### Fault-Tolerant & Privacy-Preserving Design
- **Zero Personal Identifiers**: No IP addresses, cookies, browser fingerprints, or user identifiers are ever recorded.
- **Fault Isolation**: Analytics recording runs in a safe failure-isolated block. If the SQLite database becomes locked or unavailable, search operations continue to return results seamlessly.

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
   - Analytics REST API: `http://127.0.0.1:5000/api/analytics/summary`

4. **Run the Automated Performance Benchmark Suite:**
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
10. ✅ **Stage 10** — Search Analytics & Performance Monitoring (current)
11. Stage 11 — Search Engine Optimization & Index Optimization
