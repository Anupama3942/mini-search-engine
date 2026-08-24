# Mini Search Engine

A simple command-line search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals and the basic concepts behind how search engines work.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- **Search Ranking (Term Frequency)**
- Handles basic punctuation and case-insensitivity
- Tie-breaking logic for results with identical scores
- Graceful error handling

## Technologies

- Python 3
- `pathlib`, `string`, `collections` (Python standard library)

No external packages or frameworks are required.

## Stage 4 — Search Ranking

What is search ranking? In earlier stages, the engine simply found matching documents and displayed them in alphabetical order. But if a user searches for `"python programming"`, a document that mentions "python" 5 times is likely more relevant than a document that mentions it only once. **Ranking** ensures the "best matching documents" appear at the top.

### Ranking Method

We use a simple **Term Frequency** (TF) scoring model:
`Score = Sum of frequencies of query terms in the document`

**Example:**
* **Query:** `python programming`
* **Document:** `"Python Python programming"`
* **Score:** `python` (2) + `programming` (1) = **3**

### Sorting

After calculating scores, we sort the results from highest score to lowest score using Python's built-in `sorted()` function.
If two documents have the same score, they are sorted alphabetically by their filename (deterministic tie-breaking).

### Limitations
This simple frequency-based ranking is imperfect. For example, a document that repeats the word "Python" 100 times but provides no actual value will outrank a highly informative article that mentions "Python" only 5 times. Furthermore, it treats rare words and common words as equally important.

## Next Stage
The next stage will introduce **TF-IDF** (Term Frequency - Inverse Document Frequency) to improve relevance by weighting rare words more heavily than common words.

## Project Structure

```
mini-search-engine/
│
├── documents/          # Folder containing .txt files to search through
│   ├── python.txt
│   ├── java.txt
│   ├── database.txt
│   ├── networking.txt
│   └── web.txt
│
├── tests/              # Unit tests
│   └── test_search.py
│
├── search.py           # Main search engine script
├── README.md           # This file
└── .gitignore          # Files and folders Git should ignore
```

## How to Run

Make sure you have Python 3 installed, then run:

```bash
python search.py
```

To run the unit tests:
```bash
python -m unittest tests/test_search.py
```

## Example

```
==============================
  MINI SEARCH ENGINE (Stage 4)
==============================

Documents loaded: 5
Unique terms indexed: 270

Enter search term: python programming

Searching...

Results found: 3

  1. python.txt
     Relevance Score: 3
  2. web.txt
     Relevance Score: 2
  3. java.txt
     Relevance Score: 1
```

## Future Roadmap

This is a multi-stage project:

1. ✅ **Stage 1** — Basic document search
2. ✅ **Stage 2** — Inverted Index 
3. ✅ **Stage 3** — Text Processing
4. ✅ **Stage 4** — Search Ranking (current)
5. Stage 5 — TF-IDF scoring
6. Stage 6 — Search snippets and highlighting
7. Stage 7 — Advanced queries (AND, OR, NOT)
8. Stage 8 — Web interface
9. Stage 9 — Database integration
10. Stage 10 — Testing and deployment
