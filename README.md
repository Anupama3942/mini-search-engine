# Mini Search Engine

A command-line and web-based search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals, Information Retrieval, and Web Development.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- Boolean Search (`AND`, `OR`, `NOT`, `( )`)
- **Phrase Search (`"exact phrase"`)**
- Search Ranking (TF-IDF)
- Web Interface (Flask & HTML/CSS)
- Snippet generation and query term highlighting

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re` (Python standard library)

## Stage 8 — Phrase Search

In Stage 8, the engine was upgraded to support exact phrase matching by introducing a **Positional Inverted Index**.

### Why a Positional Index?
A normal inverted index (Stage 2) tells you *which* documents contain a word, but not *where* the word is located inside that document. Because phrase searching requires words to be adjacent and in a specific order, the engine now records the sequence position of every word during indexing.

### How Phrase Search Works
When searching for `"python programming"`:
1. The engine checks the index for `python` and `programming`.
2. It filters down to documents that contain **both** words.
3. For each candidate document, it inspects the positions. It verifies that a position of `programming` is exactly `+1` from a position of `python`. 
4. If they are consecutive and in the correct order, the phrase matches!

*Note: Positional checking occurs against the normalized text tokens (after punctuation and stop words are removed).*

### Application Architecture

```text
                    User Query
                         ↓
                Query Tokenizer
                         ↓
                  Query Parser
                         ↓
             ┌───────────┴───────────┐
             ↓                       ↓
          TERM                    PHRASE
             ↓                       ↓
     Inverted Index        Positional Index
             ↓                       ↓
             └───────────┬───────────┘
                         ↓
                 Boolean Evaluation
                         ↓
                Matching Documents
                         ↓
                  TF-IDF Ranking
                         ↓
                    Web Results
```

### Advanced Boolean & Phrase Combinations
Phrase searches behave as a single boolean entity, meaning they can be perfectly combined with advanced operators:
* `"machine learning" AND python`
* `"data science" OR "machine learning"`
* `("deep learning" OR "neural networks") AND NOT java`

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
   python -m unittest tests/test_app.py tests/test_search.py
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
8. ✅ **Stage 8** — Phrase Search (current)
9. Stage 9 — Fuzzy Search & Typo Tolerance
10. Stage 10 — Database Integration
