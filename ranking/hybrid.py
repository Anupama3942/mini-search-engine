"""
Mini Search Engine - Stage 15
Hybrid (Sparse + Dense) Ranking Strategy
"""

from typing import List, Dict, Any, Optional
import config
from .base import BaseRanker
from .bm25 import BM25Ranker
from semantic.hybrid import HybridSearchEngine


class HybridRanker(BaseRanker):
    """Hybrid Retrieval combining Sparse BM25 and Dense Semantic Similarity."""

    def __init__(
        self, 
        alpha: float = config.HYBRID_ALPHA,
        hybrid_engine: Optional[HybridSearchEngine] = None
    ):
        config.validate_hybrid_params(alpha)
        self.alpha = float(alpha)
        self.hybrid_engine = hybrid_engine or HybridSearchEngine()
        self.bm25_fallback = BM25Ranker(k1=config.BM25_K1, b=config.BM25_B)

    @property
    def name(self) -> str:
        return "hybrid"

    def score(self, query_terms: List[str], doc_id: str, context: Any) -> float:
        # Hybrid scores depend on collection score normalization across all candidates
        query_text = " ".join(query_terms)
        res = self.hybrid_engine.search_hybrid(query_text, context, alpha=self.alpha, top_k=50)
        for item in res:
            if item["filename"] == doc_id:
                return float(item["score"])
        return 0.0

    def rank(
        self, 
        query_terms: List[str], 
        candidate_docs: Any, 
        context: Any, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        query_text = " ".join(query_terms)
        results = self.hybrid_engine.search_hybrid(query_text, context, alpha=self.alpha, top_k=top_k)
        if not results:
            return self.bm25_fallback.rank(query_terms, candidate_docs, context, top_k=top_k)
        return results

    def explain_score(self, query_terms: List[str], doc_id: str, context: Any) -> Dict[str, Any]:
        query_text = " ".join(query_terms)
        results = self.hybrid_engine.search_hybrid(query_text, context, alpha=self.alpha, top_k=50)
        
        target = None
        for item in results:
            if item["filename"] == doc_id:
                target = item
                break

        if not target:
            return {
                "algorithm": self.name,
                "doc_id": doc_id,
                "alpha": self.alpha,
                "status": "not_in_candidate_union",
                "total_score": 0.0
            }

        return {
            "algorithm": self.name,
            "doc_id": doc_id,
            "alpha": self.alpha,
            "bm25_raw_score": target["bm25_raw_score"],
            "semantic_raw_score": target["semantic_raw_score"],
            "bm25_normalized": target["bm25_normalized"],
            "semantic_normalized": target["semantic_normalized"],
            "formula": f"({self.alpha} * {target['bm25_normalized']}) + ({1.0 - self.alpha} * {target['semantic_normalized']})",
            "total_score": target["score"]
        }
