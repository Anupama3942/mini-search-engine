# Mini Search Engine

A simple command-line search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals and the basic concepts behind how search engines work.

## Features

- Search through multiple text documents
- Case-insensitive search
- Handles basic punctuation
- Multiple document support
- Result counting and numbering
- Simple CLI (Command-Line Interface)
- Fast querying using an Inverted Index
- Graceful error handling

## Technologies

- Python 3
- `pathlib` (Python standard library)
- `string` (Python standard library)

No external packages or frameworks are required.

## Stage 2 — Inverted Index

In Stage 1, the search engine checked every single document from beginning to end whenever a user performed a search. That is very slow! 

In Stage 2, we introduced an **Inverted Index**. Instead of scanning documents during a search, we pre-process the documents once to build a dictionary that maps each unique word to a set of documents containing that word.

**Why is it useful?** Searching becomes almost instant, no matter how many documents you have. A dictionary lookup is significantly faster on average than a linear scan through every file's text.

### How it Works

```
Documents
   ↓
Tokenization (Split into words)
   ↓
Normalization (Lowercase, remove punctuation)
   ↓
Inverted Index Building (Map words to documents)
   ↓
User Query (e.g. "python")
   ↓
Dictionary Lookup (O(1) average time)
   ↓
Matching Documents
```

### Important Data Structures Used

* **Dictionary (`dict`)**: Maps a word (string) to the documents that contain it. 
  Example: `"python" -> {"python.txt", "web.txt"}`
* **Set (`set`)**: Holds the unique collection of document filenames for a given word. It prevents duplicates so a document isn't listed 5 times if a word appears 5 times in it.
* **List (`list`)**: Used to return and sort the final ordered search results.

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
├── search.py           # Main search engine script
├── README.md           # This file
└── .gitignore          # Files and folders Git should ignore
```

## How to Run

Make sure you have Python 3 installed, then run:

```bash
python search.py
```

## Example

```
==============================
  MINI SEARCH ENGINE
==============================

5 documents loaded.
Inverted index created.
Unique words indexed: 288

Enter search term: python

Searching...

Results found: 2

  1. python.txt
  2. web.txt
```

## Future Roadmap

This is a multi-stage project:

1. ✅ **Stage 1** — Basic document search
2. ✅ **Stage 2** — Inverted Index (current)
3. Stage 3 — Text preprocessing (stopwords, stemming)
4. Stage 4 — Better search and ranking
5. Stage 5 — TF-IDF scoring
6. Stage 6 — Search snippets and highlighting
7. Stage 7 — Advanced queries (AND, OR, NOT)
8. Stage 8 — Web interface
9. Stage 9 — Database integration
10. Stage 10 — Testing and deployment
