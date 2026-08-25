"""
Mini Search Engine - Central Configuration
Stage 11: Search Engine & Index Optimization
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
TOP_K_DEFAULT = 50
EARLY_TERMINATION_ENABLED = True
POSTING_LIST_SORTING = True

# Analytics & Observability
ANALYTICS_ENABLED = True
BENCHMARK_MODE = False
