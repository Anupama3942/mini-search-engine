"""
Mini Search Engine - Central Configuration
Stage 16: Productionization, Advanced Retrieval Architecture & Deployment
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "documents"))
DOCUMENTS_DIR = DATA_DIR
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
ANALYTICS_DB_PATH = Path(os.getenv("ANALYTICS_DB_PATH", BASE_DIR / "analytics.db"))
INDEX_CACHE_PATH = Path(os.getenv("INDEX_CACHE_PATH", BASE_DIR / "index_cache.json"))

# LTR Model Paths
LTR_MODEL_PATH = MODELS_DIR / "ltr_model.json"
LTR_METADATA_PATH = MODELS_DIR / "ltr_metadata.json"

# Vector Store Paths (Stage 15)
VECTOR_INDEX_PATH = MODELS_DIR / "vector_index.json"
VECTOR_METADATA_PATH = MODELS_DIR / "vector_index_metadata.json"

# Application Environment Profile
APP_ENV = os.getenv("APP_ENV", "development").lower().strip()  # "development", "testing", "production"
IS_PRODUCTION = (APP_ENV == "production")
DEBUG = os.getenv("DEBUG", "false" if IS_PRODUCTION else "true").lower() in ("true", "1", "yes")
HOST = os.getenv("HOST", "0.0.0.0" if IS_PRODUCTION else "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if IS_PRODUCTION else "DEBUG").upper()
LOG_QUERIES = os.getenv("LOG_QUERIES", "false" if IS_PRODUCTION else "true").lower() in ("true", "1", "yes")

# Request & Rate Limiting
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 500))
MIN_TOP_K = 1
MAX_TOP_K = int(os.getenv("MAX_TOP_K", 100))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 10))
TOP_K_DEFAULT = DEFAULT_TOP_K
DEFAULT_PAGE_SIZE = int(os.getenv("DEFAULT_PAGE_SIZE", 10))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 120))  # requests per minute per IP
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))

# Search & Ranking Configuration
DEFAULT_RANKING_ALGORITHM = os.getenv("DEFAULT_RANKING_ALGORITHM", "bm25").lower()
CANDIDATE_POOL_SIZE = int(os.getenv("CANDIDATE_POOL_SIZE", 50))
SEARCH_TIMEOUT_SECONDS = float(os.getenv("SEARCH_TIMEOUT_SECONDS", 2.0))
EARLY_TERMINATION_ENABLED = os.getenv("EARLY_TERMINATION_ENABLED", "true").lower() in ("true", "1", "yes")
POSTING_LIST_SORTING = True

# Feature Flags
ENABLE_SEMANTIC = os.getenv("ENABLE_SEMANTIC", "true").lower() in ("true", "1", "yes")
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() in ("true", "1", "yes")
ENABLE_LTR = os.getenv("ENABLE_LTR", "true").lower() in ("true", "1", "yes")
ENABLE_ADVANCED_PIPELINES = os.getenv("ENABLE_ADVANCED_PIPELINES", "true").lower() in ("true", "1", "yes")

# Caching Configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", 256))
FUZZY_CACHE_SIZE = int(os.getenv("FUZZY_CACHE_SIZE", 512))
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 512))
IDF_CACHE_ENABLED = True

# BM25 Hyperparameters
BM25_K1 = float(os.getenv("BM25_K1", 1.2))
BM25_B = float(os.getenv("BM25_B", 0.75))

def validate_bm25_params(k1: float, b: float) -> None:
    """Validate BM25 hyperparameter bounds."""
    if k1 <= 0:
        raise ValueError(f"Invalid BM25 parameter k1={k1}. Must be > 0.")
    if not (0.0 <= b <= 1.0):
        raise ValueError(f"Invalid BM25 parameter b={b}. Must be between 0.0 and 1.0.")

# Learning-to-Rank (LTR) Configuration
FEATURE_VERSION = "1.0"
LTR_DEFAULT_REGULARIZATION_C = float(os.getenv("LTR_REGULARIZATION_C", 0.01))
LTR_LEARNING_RATE = float(os.getenv("LTR_LEARNING_RATE", 0.1))
LTR_EPOCHS = int(os.getenv("LTR_EPOCHS", 1000))

# Semantic & Vector Search Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 64))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))

def validate_hybrid_params(alpha: float) -> None:
    """Validate hybrid retrieval alpha parameter."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Invalid hybrid alpha={alpha}. Must be between 0.0 and 1.0.")

# Security & CORS
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# Analytics & Observability
ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "true").lower() in ("true", "1", "yes")
BENCHMARK_MODE = False
