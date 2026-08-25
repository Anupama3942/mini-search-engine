"""
Mini Search Engine - Stage 13
Base Ranking Strategy Interface
"""

from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any, Optional
import heapq


class BaseRanker(ABC):
    """Abstract Base Class for all ranking algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm identifier string."""
        pass

    @abstractmethod
    def score(self, query_terms: List[str], doc_id: str, context: Any) -> float:
        """Calculate the relevance score for a document given query terms and engine context."""
        pass

    def rank(
        self, 
        query_terms: List[str], 
        candidate_docs: Set[str], 
        context: Any, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate documents using the scoring method.
        Returns a list of dicts: [{"filename": doc_id, "score": score, ...}, ...]
        sorted by descending score.
        """
        if not candidate_docs:
            return []

        results = []
        for doc_id in candidate_docs:
            doc_score = self.score(query_terms, doc_id, context)
            results.append({
                "filename": doc_id,
                "score": round(doc_score, 6),
                "ranking_algorithm": self.name
            })

        # Sort descending by score, tie-breaking by filename ascending
        if top_k and len(results) > top_k:
            ranked = heapq.nsmallest(
                top_k, 
                results, 
                key=lambda x: (-x["score"], x["filename"])
            )
        else:
            ranked = sorted(results, key=lambda x: (-x["score"], x["filename"]))

        return ranked

    @abstractmethod
    def explain_score(self, query_terms: List[str], doc_id: str, context: Any) -> Dict[str, Any]:
        """Provide detailed human-readable score attribution breakdown for debugging."""
        pass
