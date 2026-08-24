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
- Text processing pipeline (Stop-word removal, Normalization, Tokenization)
- Graceful error handling

## Technologies

- Python 3
- `pathlib` (Python standard library)
- `string` (Python standard library)

No external packages or frameworks are required.

## Stage 3 — Text Processing

What is text processing? Raw text in documents often contains punctuation, capital letters, and very common words (like "the", "is", "a") that aren't useful for searching. Before adding text to our Inverted Index, we "clean" it. This makes the index much more efficient and allows searches like "PYTHON!" and "python" to match perfectly.

### Processing Pipeline

When reading documents (and when searching queries), we run the text through this pipeline:

```
Raw Text
   ↓
Lowercase
   ↓
Punctuation Removal
   ↓
Tokenization (Splitting into individual words)
   ↓
Stop Word Removal (Removing words like "is", "a", "the")
   ↓
Normalized Tokens
   ↓
Inverted Index (or Search Engine Lookup)
```

**Example:**
- **Input:** `"Python, is a powerful programming language!"`
- **Output:** `["python", "powerful", "programming", "language"]`

### Stop Words
Stop words are extremely common words (e.g., "a", "an", "the", "is", "in") that appear in almost every English document. Because they are so common, they aren't very helpful for finding specific information. By filtering them out, we save space in our index and speed up our searches!

### What We Are NOT Doing Yet
- **Stemming:** We are not reducing words to their root form (e.g., "programming" -> "program").
- **Lemmatization:** We are not using dictionary-based morphological analysis.
- **TF-IDF:** We are not scoring words based on how rare or important they are.
- **Ranking:** The results are just sorted alphabetically, not by relevance.
- **NLP Libraries:** Everything is still built with plain Python string operations!

### Handling Numbers
For this stage, numbers (e.g., "3" in "Python 3") remain as standard tokens. They are not stripped out, allowing simple exact-match searches for numbers.

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
  MINI SEARCH ENGINE (Stage 3)
==============================

6 documents loaded.
Inverted index created with Text Processing.
Unique processed words indexed: 270

Enter search term (or type '--debug' to turn on processing demo): --debug

[Debug Mode Enabled]
Enter search term: Python, is a powerful programming language!

Searching...

[DEBUG] Original: 'Python, is a powerful programming language!'
[DEBUG] Without punctuation: 'python  is a powerful programming language '
[DEBUG] Tokens: ['python', 'is', 'a', 'powerful', 'programming', 'language']
[DEBUG] After stop words: ['python', 'powerful', 'programming', 'language']

Results found: 6

  1. database.txt
  2. java.txt
  ...
```

## Future Roadmap

This is a multi-stage project:

1. ✅ **Stage 1** — Basic document search
2. ✅ **Stage 2** — Inverted Index 
3. ✅ **Stage 3** — Text Processing (current)
4. Stage 4 — Better search and ranking
5. Stage 5 — TF-IDF scoring
6. Stage 6 — Search snippets and highlighting
7. Stage 7 — Advanced queries (AND, OR, NOT)
8. Stage 8 — Web interface
9. Stage 9 — Database integration
10. Stage 10 — Testing and deployment
