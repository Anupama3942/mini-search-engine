"""
Mini Search Engine - Stage 13 Ranking Package
"""

from typing import Optional
from .base import BaseRanker
from .frequency import FrequencyRanker
from .tfidf import TFIDFRanker
from .bm25 import BM25Ranker
import config


def get_ranker(
    algorithm: Optional[str] = None, 
    k1: Optional[float] = None, 
    b: Optional[float] = None
) -> BaseRanker:
    """
    Factory function to instantiate ranking strategies.
    Supported algorithms: "bm25", "tfidf", "frequency".
    """
    algo_name = (algorithm or config.DEFAULT_RANKING_ALGORITHM).lower().strip()
    
    if algo_name == "bm25":
        param_k1 = k1 if k1 is not None else config.BM25_K1
        param_b = b if b is not None else config.BM25_B
        return BM25Ranker(k1=param_k1, b=param_b)
    elif algo_name in ("tfidf", "tf-idf"):
        return TFIDFRanker()
    elif algo_name in ("frequency", "tf", "freq"):
        return FrequencyRanker()
    else:
        raise ValueError(f"Unknown ranking algorithm '{algo_name}'. Choose from: 'bm25', 'tfidf', 'frequency'.")


__all__ = ["BaseRanker", "FrequencyRanker", "TFIDFRanker", "BM25Ranker", "get_ranker"]
