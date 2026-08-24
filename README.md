# Mini Search Engine

A simple command-line search engine built with Python. This project searches through `.txt` documents stored in a local folder and returns matching filenames.

Built as a learning project to understand Python fundamentals and the basic concepts behind how search engines work.

## Features

- Search through multiple text documents
- Case-insensitive search
- Multiple document support
- Result counting and numbering
- Simple CLI (Command-Line Interface)
- Graceful error handling

## Technologies

- Python 3
- `pathlib` (Python standard library)

No external packages or frameworks are required.

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

| File / Folder   | Purpose                                              |
|-----------------|------------------------------------------------------|
| `documents/`    | Holds all the `.txt` files that the engine searches   |
| `search.py`     | The main Python program — loads files and runs search |
| `README.md`     | Project documentation (you are reading it)            |
| `.gitignore`    | Tells Git which files to exclude from version control |

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

Enter search term: python

Searching...

Results found: 2

  1. python.txt
  2. web.txt
```

## Concepts Learned

Building this project covers the following Python and CS fundamentals:

| Concept             | How It's Used                                                  |
|---------------------|----------------------------------------------------------------|
| **Variables**       | Storing the query, filenames, and file contents                |
| **Lists**           | Collecting matching filenames in the search results            |
| **Dictionaries**    | Mapping each filename to its content for fast lookup           |
| **Functions**       | Organizing code into `load_documents()`, `search_documents()`, `main()` |
| **Loops**           | Iterating over files and documents to read and search          |
| **Conditions**      | Checking for empty input, missing directories, and matches     |
| **File Handling**   | Reading `.txt` files safely using `with open()`                |
| **pathlib**         | Finding the documents folder and listing `.txt` files          |
| **String Operations** | `.lower()` for case-insensitive search, `.strip()` for input cleaning, `in` for substring matching |
| **Searching**       | Linear search — checking every document for the query term     |

## Future Roadmap

This is Stage 1 of a multi-stage project:

1. ✅ **Stage 1** — Basic document search (current)
2. Stage 2 — Inverted Index
3. Stage 3 — Text preprocessing (stopwords, stemming)
4. Stage 4 — Better search and ranking
5. Stage 5 — TF-IDF scoring
6. Stage 6 — Search snippets and highlighting
7. Stage 7 — Advanced queries (AND, OR, NOT)
8. Stage 8 — Web interface
9. Stage 9 — Database integration
10. Stage 10 — Testing and deployment
