# Mini Search Engine

A command-line and web-based search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals, Information Retrieval, and Web Development.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- **Search Ranking (TF-IDF)**
- **Web Interface (Flask & HTML/CSS)**
- Snippet generation and query term highlighting
- Handles basic punctuation and case-insensitivity
- Tie-breaking logic for results with identical scores
- Graceful error handling

## Technologies

- Python 3
- Flask (Web Framework)
- Jinja2 (Templating)
- HTML5 & CSS3
- `pathlib`, `string`, `collections`, `math`, `html`, `re` (Python standard library)

## Stage 6 — Web Interface

In Stage 6, the command-line search engine was upgraded to a fully functional local web application using the **Flask** microframework. 

### Why Flask?
Flask is a lightweight Python web framework. It acts as the bridge between a user's web browser and our existing Python search engine. Instead of rewriting the search logic in JavaScript, Flask listens for HTTP requests, passes the user's query to the `SearchEngine` class, and then injects the ranked results into an HTML template to send back to the browser.

### Application Architecture

```text
       User
        ↓
    Web Browser
        ↓
      Flask (app.py)
        ↓
    Search Engine (search.py)
        ↓
  Text Processing
        ↓
  Inverted Index
        ↓
  TF-IDF Ranking
        ↓
  Ranked Results
        ↓
  Flask Template (results.html)
        ↓
    Web Browser
```

### Separation of Concerns
The core search logic was refactored into a `SearchEngine` class, but it remains entirely independent of Flask. This means you can still run `python search.py` to use the CLI version, or `python app.py` to use the Web version. Both use the exact same TF-IDF ranking logic.

### Snippets & Highlighting
When a document matches, the search engine extracts the first 200 characters of the document to serve as a snippet. It uses the `html` library to safely escape the text (preventing Cross-Site Scripting / XSS attacks), and then uses Regular Expressions (`re`) to wrap the matching query terms in HTML `<mark>` tags for visual highlighting.

## Project Structure

```text
mini-search-engine/
│
├── app.py              # Flask Web Application
├── search.py           # Core Search Engine & CLI
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

## Example Search Flow
1. **Search:** You enter `python programming` in the search box on the homepage and submit.
2. **HTTP GET:** The browser sends a `GET` request to `/search?q=python+programming`.
3. **Flask:** `app.py` receives the request, extracts the `q` parameter, and calls `search_engine.search("python programming")`.
4. **Search Engine:** The TF-IDF ranker processes the query, identifies matching documents using the inverted index, calculates scores, generates snippets, and returns a sorted list of dictionaries.
5. **Template Rendering:** Flask passes this list to `results.html`, which loops through the results to generate the final HTML page.
6. **Browser:** The user sees a beautifully styled list of ranked search results.

## Future Roadmap

This is a multi-stage project:

1. ✅ **Stage 1** — Basic document search
2. ✅ **Stage 2** — Inverted Index 
3. ✅ **Stage 3** — Text Processing
4. ✅ **Stage 4** — Search Ranking
5. ✅ **Stage 5** — TF-IDF Ranking
6. ✅ **Stage 6** — Web Interface (current)
7. Stage 7 — Advanced queries (AND, OR, NOT)
8. Stage 8 — Database integration
9. Stage 9 — Pagination & Caching
10. Stage 10 — Deployment
