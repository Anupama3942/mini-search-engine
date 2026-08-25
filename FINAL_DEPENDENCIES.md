# Dependency Review

## External Dependencies

| Dependency | Version | Purpose | Required? | Dev Only? |
| --- | --- | --- | --- | --- |
| Flask | 3.0.3 | Web framework, REST API, template rendering | Yes | No |

## Python Standard Library Modules Used

The project relies heavily on the Python standard library to remain lightweight and dependency-free where possible:

- `sqlite3`: Analytics and logging
- `hashlib`: A/B testing hashing
- `math`: BM25 and TF-IDF calculations
- `collections`: Inverted index and posting lists
- `json`: Serialization for API and storage
- `tracemalloc`: Memory profiling
- `unittest`: Testing framework
- `pathlib`: Path manipulation
- `logging`: Structured logs
- `time`: Performance timing
- `uuid`: Request ID generation
- `unicodedata`: Text normalization
- `re`: Query parsing and tokenization
- `os`: Environment configuration
- `functools`: Caching (`lru_cache`)

> [!NOTE]
> The project uses **no external ML libraries** (no `numpy`, `sklearn`, `torch`, etc.). Everything is implemented in pure Python.

## Recommendations for Production

- Recommend adding a production WSGI server like **Gunicorn** or **Waitress** for deployment.
