"""
Mini Search Engine - Stage 13, 14 & 15 Ranking Package
"""

from typing import Optional
from .base import BaseRanker
from .frequency import FrequencyRanker
from .tfidf import TFIDFRanker
from .bm25 import BM25Ranker
from .ltr import LTRRanker
from .semantic import SemanticRanker
from .hybrid import HybridRanker
import config


def get_ranker(
    algorithm: Optional[str] = None, 
    k1: Optional[float] = None, 
    b: Optional[float] = None,
    alpha: Optional[float] = None
) -> BaseRanker:
    """
    Factory function to instantiate ranking strategies.
    Supported algorithms: "bm25", "tfidf", "frequency", "ltr", "semantic", "hybrid".
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
    elif algo_name in ("ltr", "learning-to-rank", "learning_to_rank"):
        return LTRRanker()
    elif algo_name in ("semantic", "vector", "dense"):
        return SemanticRanker()
    elif algo_name in ("hybrid", "dense_sparse"):
        param_alpha = alpha if alpha is not None else config.HYBRID_ALPHA
        return HybridRanker(alpha=param_alpha)
    else:
        raise ValueError(f"Unknown ranking algorithm '{algo_name}'. Choose from: 'bm25', 'tfidf', 'frequency', 'ltr', 'semantic', 'hybrid'.")


__all__ = [
    "BaseRanker", 
    "FrequencyRanker", 
    "TFIDFRanker", 
    "BM25Ranker", 
    "LTRRanker", 
    "SemanticRanker", 
    "HybridRanker", 
    "get_ranker"
]
