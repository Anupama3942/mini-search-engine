"""
Mini Search Engine - Stage 13
Term Frequency Ranker (Baseline Strategy)
"""

from typing import List, Dict, Any
from .base import BaseRanker


class FrequencyRanker(BaseRanker):
    """Ranks documents purely by the raw count of query terms appearing in the document."""

    @property
    def name(self) -> str:
        return "frequency"

    def score(self, query_terms: List[str], doc_id: str, context: Any) -> float:
        counts = context.term_counts.get(doc_id, {})
        total_freq = 0
        for term in set(query_terms):
            total_freq += counts.get(term, 0)
        return float(total_freq)

    def explain_score(self, query_terms: List[str], doc_id: str, context: Any) -> Dict[str, Any]:
        counts = context.term_counts.get(doc_id, {})
        term_breakdown = {}
        total = 0
        for term in set(query_terms):
            cnt = counts.get(term, 0)
            term_breakdown[term] = {"term_frequency": cnt, "contribution": float(cnt)}
            total += cnt

        return {
            "algorithm": self.name,
            "doc_id": doc_id,
            "total_score": float(total),
            "term_contributions": term_breakdown
        }
