"""
Mini Search Engine - Central Configuration
Stage 14: Learning-to-Rank (LTR) & Advanced Ranking Configuration
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
ANALYTICS_DB_PATH = BASE_DIR / "analytics.db"
INDEX_CACHE_PATH = BASE_DIR / "index_cache.json"
MODELS_DIR = BASE_DIR / "models"
LTR_MODEL_PATH = MODELS_DIR / "ltr_model.json"
LTR_METADATA_PATH = MODELS_DIR / "ltr_metadata.json"

# Caching Configuration
CACHE_ENABLED = True
QUERY_CACHE_SIZE = 256
FUZZY_CACHE_SIZE = 512
IDF_CACHE_ENABLED = True

# Search & Ranking Configuration
DEFAULT_RANKING_ALGORITHM = "bm25"  # Options: "bm25", "tfidf", "frequency", "ltr"
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

# Learning-to-Rank (LTR) Configuration
FEATURE_VERSION = "1.0"
LTR_DEFAULT_REGULARIZATION_C = 1.0
LTR_LEARNING_RATE = 0.1
LTR_EPOCHS = 1000

# Analytics & Observability
ANALYTICS_ENABLED = True
BENCHMARK_MODE = False
