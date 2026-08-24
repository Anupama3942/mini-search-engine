# Mini Search Engine

A simple command-line search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals and the basic concepts behind how search engines work.

## Features

- Search through multiple text documents
- Fast querying using an Inverted Index
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- **Search Ranking (TF-IDF)**
- Handles basic punctuation and case-insensitivity
- Tie-breaking logic for results with identical scores
- Graceful error handling

## Technologies

- Python 3
- `pathlib`, `string`, `collections`, `math` (Python standard library)

No external packages or frameworks are required.

## Stage 5 — TF-IDF Ranking

Why is TF-IDF introduced? In Stage 4, ranking was based merely on how many times a term appeared. If a user searched for "python programming", a document that repeated the word "programming" 500 times would win, even if another document perfectly balanced rare and common words. TF-IDF fixes this by punishing common words and rewarding rare words.

### 1. Term Frequency (TF)
`TF(t, d) = count(t, d) / total_terms(d)`
How often does the term `t` appear in the document `d`? We divide by `total_terms` so that longer documents don't get an unfair advantage.

### 2. Document Frequency (DF)
`DF(t) = number of documents containing t`
How many documents does the term `t` appear in? Note that this counts *documents*, not total occurrences.

### 3. Inverse Document Frequency (IDF)
`IDF(t) = log(N / DF(t))`
This is the magic. If `N` (total documents) is 100, and "programming" appears in 95 of them, its IDF is `log(100/95) = 0.05` (very small). If "tensorflow" appears in 3 documents, its IDF is `log(100/3) = 3.5` (very large). Rare words get a higher weight.

### 4. TF-IDF
`TF-IDF(t, d) = TF(t, d) * IDF(t)`
The final score for a term in a document.

### 5. Document Score
`Score(d, q) = Σ TF-IDF(t, d)`
The sum of the TF-IDF scores for each query term in the document.

### Worked Example
Imagine 3 documents (`N=3`):
* `Doc1`: `"python programming python"`
* `Doc2`: `"java programming"`
* `Doc3`: `"python web"`

For `"python"` in `Doc1`:
* **TF:** "python" appears 2 times out of 3 total words -> `2 / 3 = 0.666`
* **DF:** "python" appears in 2 documents (`Doc1` and `Doc3`) -> `2`
* **IDF:** `log(3 / 2) = 0.405`
* **TF-IDF:** `0.666 * 0.405 = 0.270`

### Stage 4 vs Stage 5
* **Stage 4:** Counted word frequencies directly. Highly vulnerable to keyword stuffing and over-valuing common words.
* **Stage 5:** Balances word frequencies with word rarity. A term is only powerful if it appears frequently in a document *and* rarely across the entire document collection.

### Limitations of TF-IDF
TF-IDF is a powerful statistical measure but it doesn't understand context or meaning. "Apple" (the fruit) and "Apple" (the company) are treated identically. It also doesn't consider typos or synonyms. Modern search uses embeddings and neural ranking alongside statistical measures to solve this.

## Project Structure

```
mini-search-engine/
│
├── documents/          # Folder containing .txt files to search through
├── tests/              # Unit tests
│   └── test_search.py
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
  MINI SEARCH ENGINE (Stage 5)
==============================

Documents loaded: 6
Unique terms indexed: 270

Enter search term: python programming

Searching...

Results found: 4

  1. python.txt
     TF-IDF Score: 0.1764
  2. web.txt
     TF-IDF Score: 0.1171
  3. database.txt
     TF-IDF Score: 0.0504
  4. java.txt
     TF-IDF Score: 0.0336
```

## Future Roadmap

This is a multi-stage project:

1. ✅ **Stage 1** — Basic document search
2. ✅ **Stage 2** — Inverted Index 
3. ✅ **Stage 3** — Text Processing
4. ✅ **Stage 4** — Search Ranking
5. ✅ **Stage 5** — TF-IDF Ranking (current)
6. Stage 6 — Search snippets and highlighting
7. Stage 7 — Advanced queries (AND, OR, NOT)
8. Stage 8 — Web interface
9. Stage 9 — Database integration
10. Stage 10 — Testing and deployment
