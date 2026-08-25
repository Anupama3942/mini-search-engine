# Mini Search Engine

A command-line and web-based search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals, Information Retrieval, and Web Development.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index
- Positional Index for exact phrase matching (`"exact phrase"`)
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- Boolean Search (`AND`, `OR`, `NOT`, `( )`)
- **Fuzzy Search & Typo Tolerance (Levenshtein Distance via Dynamic Programming)**
- Search Ranking (TF-IDF)
- Web Interface (Flask & HTML/CSS)
- Snippet generation and query term highlighting
- Graceful error handling and XSS protection

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re` (Python standard library)

## Stage 9 — Fuzzy Search & Typo Tolerance

In Stage 9, the search engine was upgraded with **Typo Tolerance** using the **Levenshtein Distance** algorithm implemented from scratch with **Dynamic Programming**.

### What is Levenshtein Distance?
Levenshtein distance measures the minimum number of single-character edits required to transform one word into another:
1. **Insertion**: inserting a missing character (e.g., `pythn` &rarr; `python`, distance = 1)
2. **Deletion**: removing an extra character (e.g., `programmin` &rarr; `programming`, distance = 1)
3. **Substitution**: replacing an incorrect character (e.g., `jython` &rarr; `python`, distance = 1)

### Dynamic Programming Recurrence
For two strings $A$ of length $m$ and $B$ of length $n$, the DP matrix cell $dp[i][j]$ represents the edit distance between prefixes $A[0..i]$ and $B[0..j]$:

- **Base Cases**: $dp[i][0] = i$ (deletions), $dp[0][j] = j$ (insertions)
- **If characters match** ($A[i-1] == B[j-1]$): $dp[i][j] = dp[i-1][j-1]$
- **If characters differ**:
  $$dp[i][j] = 1 + \min(\text{deletion: } dp[i-1][j], \text{insertion: } dp[i][j-1], \text{substitution: } dp[i-1][j-1])$$

### Thresholds (Maximum Edit Distance)
To prevent bad or overly aggressive matches, thresholds adapt based on word length:
- **Length 1–3**: Max distance = 0 (exact match only)
- **Length 4–6**: Max distance &le; 1 (e.g., `pythn` &rarr; `python`)
- **Length 7+**: Max distance &le; 2 (e.g., `programing` &rarr; `programming`)

### Exact-Match Optimization & Vocabulary Lookup
Fuzzy matching is only executed when a term does not exist in the indexed vocabulary. If the term already exists (exact match), it is resolved instantly ($O(1)$) without invoking Levenshtein distance.

### Application Architecture

```text
                    User Query
                         ↓
                 Query Normalization
                         ↓
                  Query Tokenizer
                         ↓
                   Query Parser
                         ↓
                 Term Resolution
                         ↓
             ┌───────────┴───────────┐
             ↓                       ↓
        Exact Match            Fuzzy Match
             │                       │
             │                 Levenshtein
             │                     Distance
             │                       │
             └───────────┬───────────┘
                         ↓
                  Inverted Index
                         ↓
                 Matching Documents
                         ↓
                   TF-IDF Ranking
                         ↓
                   Search Results
```

### Algorithmic Complexity
- **Levenshtein DP Matrix**: Time $O(m \times n)$, Space $O(m \times n)$ (or $O(\min(m, n))$ with two-row optimization).
- **Vocabulary Search**: Compares query against vocabulary terms with a length-difference filter $|len(A) - len(B)| \le max\_dist$ to skip impossible candidates.
- **Fuzzy Cache**: Resolved terms are cached in memory to eliminate repeated calculations for identical typos.

### Boolean & Phrase Compatibility
- **Boolean Expressions**: Typos inside Boolean queries are seamlessly resolved before evaluation (e.g. `pythn AND programing` &rarr; `python AND programming`).
- **Phrase Searches**: Quoted phrases (e.g. `"machine learning"`) strictly preserve exact phrase matching.

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
   Open your browser and navigate to `http://127.0.0.1:5000`

4. **(Optional) Run the CLI version:**
   ```bash
   python search.py
   ```

5. **Run tests:**
   ```bash
   python -m unittest discover tests
   ```

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
9. ✅ **Stage 9** — Fuzzy Search & Typo Tolerance (current)
10. Stage 10 — Search Analytics & Performance Monitoring
