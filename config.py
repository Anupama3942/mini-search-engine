"""
Mini Search Engine - Central Configuration
Stage 15: Neural / Semantic Search & Vector Retrieval Configuration
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
ANALYTICS_DB_PATH = BASE_DIR / "analytics.db"
INDEX_CACHE_PATH = BASE_DIR / "index_cache.json"
MODELS_DIR = BASE_DIR / "models"

# LTR Model Paths
LTR_MODEL_PATH = MODELS_DIR / "ltr_model.json"
LTR_METADATA_PATH = MODELS_DIR / "ltr_metadata.json"

# Vector Store Paths (Stage 15)
VECTOR_INDEX_PATH = MODELS_DIR / "vector_index.json"
VECTOR_METADATA_PATH = MODELS_DIR / "vector_index_metadata.json"

# Caching Configuration
CACHE_ENABLED = True
QUERY_CACHE_SIZE = 256
FUZZY_CACHE_SIZE = 512
IDF_CACHE_ENABLED = True
EMBEDDING_CACHE_SIZE = 512

# Search & Ranking Configuration
DEFAULT_RANKING_ALGORITHM = "bm25"  # Options: "bm25", "tfidf", "frequency", "ltr", "semantic", "hybrid"
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

# Semantic & Vector Search Configuration (Stage 15)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 64
HYBRID_ALPHA = 0.5   # Weight for sparse BM25 in hybrid score (0 <= alpha <= 1)

def validate_hybrid_params(alpha: float) -> None:
    """Validate hybrid retrieval alpha parameter."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Invalid hybrid alpha={alpha}. Must be between 0.0 and 1.0.")

# Analytics & Observability
ANALYTICS_ENABLED = True
BENCHMARK_MODE = False
