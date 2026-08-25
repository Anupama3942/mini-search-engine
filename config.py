"""
Mini Search Engine - Central Configuration
Stage 13: Advanced Ranking & BM25 Configuration
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
ANALYTICS_DB_PATH = BASE_DIR / "analytics.db"
INDEX_CACHE_PATH = BASE_DIR / "index_cache.json"

# Caching Configuration
CACHE_ENABLED = True
QUERY_CACHE_SIZE = 256
FUZZY_CACHE_SIZE = 512
IDF_CACHE_ENABLED = True

# Search & Ranking Configuration
DEFAULT_RANKING_ALGORITHM = "bm25"  # Options: "bm25", "tfidf", "frequency"
TOP_K_DEFAULT = 50
EARLY_TERMINATION_ENABLED = True
POSTING_LIST_SORTING = True

# BM25 Parameters
BM25_K1 = 1.2    # Term frequency saturation parameter (k1 > 0)
BM25_B = 0.75    # Document length normalization parameter (0 <= b <= 1)

def validate_bm25_params(k1: float, b: float) -> None:
    """Validate BM25 hyperparameter bounds."""
    if k1 <= 0:
        raise ValueError(f"Invalid BM25 parameter k1={k1}. Must be > 0.")
    if not (0.0 <= b <= 1.0):
        raise ValueError(f"Invalid BM25 parameter b={b}. Must be between 0.0 and 1.0.")

# Analytics & Observability
ANALYTICS_ENABLED = True
BENCHMARK_MODE = False
