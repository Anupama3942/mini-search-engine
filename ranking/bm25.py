"""
Mini Search Engine - Stage 13
BM25 (Best Matching 25) Probabilistic Ranking Strategy
"""

import math
from typing import List, Dict, Any, Optional
from .base import BaseRanker
import config


class BM25Ranker(BaseRanker):
    """
    Best Matching 25 (BM25) Probabilistic Relevance Ranking.
    Incorporates non-linear term frequency saturation (k1)
    and document length normalization (b).
    """

    def __init__(self, k1: float = config.BM25_K1, b: float = config.BM25_B):
        config.validate_bm25_params(k1, b)
        self.k1 = float(k1)
        self.b = float(b)

    @property
    def name(self) -> str:
        return "bm25"

    def calculate_bm25_idf(self, term: str, context: Any) -> float:
        """
        Calculate numerically stable non-negative BM25 IDF:
        IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )
        """
        total_docs = len(context.documents)
        if total_docs == 0:
            return 0.0

        df = context.doc_freq.get(term, 0)
        if df == 0:
            return 0.0

        # Robertson-Spärck Jones non-negative formulation
        numerator = total_docs - df + 0.5
        denominator = df + 0.5
        return math.log(1.0 + (numerator / denominator))

    def score_term(
        self, 
        term: str, 
        doc_id: str, 
        context: Any, 
        avg_doc_len: float
    ) -> float:
        """Calculate single-term BM25 score contribution for a document."""
        raw_tf = context.term_counts.get(doc_id, {}).get(term, 0)
        if raw_tf <= 0:
            return 0.0

        idf = self.calculate_bm25_idf(term, context)
        doc_len = context.doc_lengths.get(doc_id, 0)
        
        # Avoid division by zero if corpus is empty or avg_doc_len is 0
        norm_len = (doc_len / avg_doc_len) if avg_doc_len > 0 else 1.0
        
        # BM25 TF Saturation and Length Normalization component
        numerator = raw_tf * (self.k1 + 1.0)
        denominator = raw_tf + self.k1 * (1.0 - self.b + (self.b * norm_len))
        
        tf_component = numerator / denominator
        return idf * tf_component

    def score(self, query_terms: List[str], doc_id: str, context: Any) -> float:
        """
        Calculate total BM25 score for a document across all unique query terms:
        BM25(D, Q) = sum( term_score(q_i, D) )
        """
        if not context.documents or doc_id not in context.documents:
            return 0.0

        avg_doc_len = context.index_stats.get("avg_document_length", 0.0)
        if avg_doc_len <= 0 and len(context.documents) > 0:
            total_tokens = sum(context.doc_lengths.values())
            avg_doc_len = total_tokens / len(context.documents)

        total_score = 0.0
        # Iterate over unique query terms to prevent unintentional multiplication
        for term in set(query_terms):
            total_score += self.score_term(term, doc_id, context, avg_doc_len)

        return total_score

    def explain_score(self, query_terms: List[str], doc_id: str, context: Any) -> Dict[str, Any]:
        """Provide detailed human-readable score attribution breakdown for debugging."""
        doc_len = context.doc_lengths.get(doc_id, 0)
        total_docs = len(context.documents)
        avg_doc_len = context.index_stats.get("avg_document_length", 0.0)
        if avg_doc_len <= 0 and total_docs > 0:
            avg_doc_len = sum(context.doc_lengths.values()) / total_docs

        term_breakdown = {}
        total_score = 0.0

        for term in set(query_terms):
            raw_tf = context.term_counts.get(doc_id, {}).get(term, 0)
            df = context.doc_freq.get(term, 0)
            idf = self.calculate_bm25_idf(term, context)
            
            norm_len = (doc_len / avg_doc_len) if avg_doc_len > 0 else 1.0
            numerator = raw_tf * (self.k1 + 1.0)
            denominator = raw_tf + self.k1 * (1.0 - self.b + (self.b * norm_len))
            tf_component = (numerator / denominator) if denominator > 0 else 0.0
            contrib = idf * tf_component

            term_breakdown[term] = {
                "raw_tf": raw_tf,
                "doc_freq": df,
                "idf": round(idf, 6),
                "tf_component": round(tf_component, 6),
                "contribution": round(contrib, 6)
            }
            total_score += contrib

        return {
            "algorithm": self.name,
            "doc_id": doc_id,
            "doc_length": doc_len,
            "avg_doc_length": round(avg_doc_len, 2),
            "k1": self.k1,
            "b": self.b,
            "total_score": round(total_score, 6),
            "term_contributions": term_breakdown
        }
