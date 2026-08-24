# Mini Search Engine

A command-line and web-based search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals, Information Retrieval, and Web Development.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- **Boolean Search (`AND`, `OR`, `NOT`, `( )`)**
- **Search Ranking (TF-IDF)**
- Web Interface (Flask & HTML/CSS)
- Snippet generation and query term highlighting
- Tie-breaking logic for results with identical scores
- Graceful error handling

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re` (Python standard library)

## Stage 7 — Boolean & Advanced Query Search

In Stage 7, the search engine was upgraded to support structured **Boolean queries**. Previously, a query like `python programming` just treated both words as positive terms to score. Now, the engine understands logical operations using Sets!

### Boolean Filtering vs. TF-IDF Ranking
It is very important to separate these two concepts:
1. **Boolean Filtering:** Determines *which* documents are allowed to appear in the results.
2. **TF-IDF Ranking:** Determines *what order* those allowed documents should be presented in.

Boolean queries act as strict filters. If you search for `python AND NOT java`, a document containing `java` will be removed from the eligible set immediately. TF-IDF will then only rank the remaining documents based on the positive term `python`.

### Supported Operators
* **`AND`**: Both terms must exist in the document.
  *(e.g., `python AND programming`)*
* **`OR`**: At least one term must exist.
  *(e.g., `python OR java`)*
* **`NOT`**: Return documents that do not contain the term.
  *(e.g., `NOT java`, `python AND NOT java`)*
* **`( )`**: Control the order of evaluation.
  *(e.g., `(python OR java) AND programming`)*

### Operator Precedence
If parentheses are not used, the engine evaluates operators in the following priority order (highest to lowest):
1. **`Parentheses ( )`**
2. **`NOT`**
3. **`AND`**
4. **`OR`**

*Example:* `python OR java AND programming` is evaluated as `python OR (java AND programming)`.

### Application Architecture

```text
       User Query
            ↓
  Boolean Query Tokenizer
            ↓
       Boolean Parser
            ↓
      Query Expression
            ↓
      Boolean Evaluation
            ↓
    Matching Document Set
            ↓
       TF-IDF Ranking
            ↓
       Ranked Results
            ↓
        Web Interface
```

* **Query Tokenization:** Splits the raw query string into words and operators while preserving parentheses.
* **Query Parser:** Uses a "Recursive Descent Parser" to read the tokens and construct an Abstract Syntax Tree (AST) representing the logic.
* **Boolean Evaluation:** Traverses the AST and uses Python Sets (`&`, `|`, `-`) against the Inverted Index to compute exactly which documents match the query.

## Project Structure

```text
mini-search-engine/
│
├── app.py              # Flask Web Application
├── search.py           # Core Search Engine & CLI
├── query_parser.py     # Boolean Tokenizer, Parser, and AST
│
├── documents/          # Folder containing .txt files
│
├── templates/          # HTML Templates for Flask
│   ├── base.html
│   ├── index.html
│   ├── results.html
│   └── 404.html
│
├── static/             # Static Assets
│   └── css/
│       └── style.css
│
├── tests/              # Unit tests
│   ├── test_app.py
│   └── test_search.py
│
├── requirements.txt
├── README.md           
└── .gitignore          
```

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
7. ✅ **Stage 7** — Boolean Search (current)
8. Stage 8 — Phrase Search
9. Stage 9 — Database integration
10. Stage 10 — Pagination & Deployment
