"""
Mini Search Engine - Stage 13
TF-IDF Ranking Strategy
"""

from typing import List, Dict, Any
from .base import BaseRanker


class TFIDFRanker(BaseRanker):
    """Ranks documents using classical TF-IDF (Normalized TF * log-IDF)."""

    @property
    def name(self) -> str:
        return "tfidf"

    def score(self, query_terms: List[str], doc_id: str, context: Any) -> float:
        total_score = 0.0
        for term in set(query_terms):
            tf = context.calculate_tf(term, doc_id)
            idf = context.calculate_idf(term)
            total_score += (tf * idf)
        return total_score

    def explain_score(self, query_terms: List[str], doc_id: str, context: Any) -> Dict[str, Any]:
        term_breakdown = {}
        total_score = 0.0
        for term in set(query_terms):
            tf = context.calculate_tf(term, doc_id)
            idf = context.calculate_idf(term)
            contrib = tf * idf
            term_breakdown[term] = {
                "tf": round(tf, 6),
                "idf": round(idf, 6),
                "contribution": round(contrib, 6)
            }
            total_score += contrib

        return {
            "algorithm": self.name,
            "doc_id": doc_id,
            "total_score": round(total_score, 6),
            "term_contributions": term_breakdown
        }
