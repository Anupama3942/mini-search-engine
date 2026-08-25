"""
Mini Search Engine - Stage 14
Learning-to-Rank (LTR) Feature Extraction & Normalization
"""

import math
from typing import List, Dict, Any, Optional, Tuple

import config
from ranking.bm25 import BM25Ranker
from ranking.tfidf import TFIDFRanker
from fuzzy_search import levenshtein_distance

FEATURE_VERSION = "1.0"

FEATURE_NAMES = [
    "bm25_score",
    "tfidf_score",
    "query_term_coverage",
    "exact_term_match_count",
    "term_frequency_sum",
    "document_length_norm",
    "query_length",
    "phrase_match",
    "title_match",
    "fuzzy_score"
]


class FeatureScaler:
    """Standard Min-Max Normalizer for continuous LTR feature vectors."""

    def __init__(self):
        self.min_vals: List[float] = []
        self.max_vals: List[float] = []
        self.fitted = False

    def fit(self, X: List[List[float]]) -> "FeatureScaler":
        if not X:
            return self
        num_features = len(X[0])
        self.min_vals = [float("inf")] * num_features
        self.max_vals = [float("-inf")] * num_features

        for row in X:
            for i, val in enumerate(row):
                if val < self.min_vals[i]:
                    self.min_vals[i] = val
                if val > self.max_vals[i]:
                    self.max_vals[i] = val

        self.fitted = True
        return self

    def transform_vector(self, vec: List[float]) -> List[float]:
        if not self.fitted or not self.min_vals:
            return list(vec)
        
        scaled = []
        for i, val in enumerate(vec):
            min_v = self.min_vals[i]
            max_v = self.max_vals[i]
            denom = max_v - min_v
            if denom > 1e-9:
                s_val = (val - min_v) / denom
            else:
                s_val = 0.0
            scaled.append(round(s_val, 6))
        return scaled

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        return [self.transform_vector(row) for row in X]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "fitted": self.fitted
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureScaler":
        scaler = cls()
        scaler.min_vals = data.get("min_vals", [])
        scaler.max_vals = data.get("max_vals", [])
        scaler.fitted = data.get("fitted", False)
        return scaler


class FeatureExtractor:
    """Extracts numerical ranking signals for a (Query, Document) pair."""

    def __init__(self):
        self.bm25_ranker = BM25Ranker(k1=config.BM25_K1, b=config.BM25_B)
        self.tfidf_ranker = TFIDFRanker()

    def extract_named_features(
        self, 
        query_terms: List[str], 
        doc_id: str, 
        context: Any,
        raw_query: str = "",
        fuzzy_corrections: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        if not context.documents or doc_id not in context.documents:
            return {name: 0.0 for name in FEATURE_NAMES}

        unique_qterms = list(set(query_terms)) if query_terms else []
        total_qterms = len(unique_qterms)
        doc_counts = context.term_counts.get(doc_id, {})
        doc_len = context.doc_lengths.get(doc_id, 0)
        avg_doc_len = context.index_stats.get("avg_document_length", 1.0)
        if avg_doc_len <= 0:
            avg_doc_len = 1.0

        # 1. BM25 Score
        bm25 = self.bm25_ranker.score(unique_qterms, doc_id, context)

        # 2. TF-IDF Score
        tfidf = self.tfidf_ranker.score(unique_qterms, doc_id, context)

        # 3. Query Term Coverage
        matched_terms = sum(1 for t in unique_qterms if doc_counts.get(t, 0) > 0)
        coverage = (matched_terms / total_qterms) if total_qterms > 0 else 0.0

        # 4. Exact Term Match Count
        exact_match_cnt = float(matched_terms)

        # 5. Term Frequency Sum
        tf_sum = float(sum(doc_counts.get(t, 0) for t in unique_qterms))

        # 6. Normalized Document Length
        doc_len_norm = doc_len / avg_doc_len

        # 7. Query Length
        q_len = float(len(query_terms))

        # 8. Phrase Match Indicator
        # Check if full query appears as a consecutive phrase in doc
        phrase_match = 0.0
        if len(unique_qterms) >= 2 and all(t in context.positional_index for t in unique_qterms):
            first_term = unique_qterms[0]
            if doc_id in context.positional_index.get(first_term, {}):
                first_positions = context.positional_index[first_term][doc_id]
                for pos in first_positions:
                    match = True
                    for offset, t in enumerate(unique_qterms[1:], start=1):
                        t_positions = context.positional_index.get(t, {}).get(doc_id, [])
                        if (pos + offset) not in t_positions:
                            match = False
                            break
                    if match:
                        phrase_match = 1.0
                        break

        # 9. Title Match Indicator
        title = doc_id.replace(".txt", "").lower()
        title_matches = sum(1 for t in unique_qterms if t in title)
        title_match = 1.0 if title_matches > 0 else 0.0

        # 10. Fuzzy Match Score
        fuzzy_score = 1.0
        if fuzzy_corrections:
            # Average similarity of corrected terms
            sims = []
            for typo, corr in fuzzy_corrections.items():
                dist = levenshtein_distance(typo.lower(), corr.lower())
                max_len = max(len(typo), len(corr), 1)
                sims.append(1.0 - (dist / max_len))
            fuzzy_score = sum(sims) / len(sims) if sims else 1.0

        return {
            "bm25_score": round(bm25, 4),
            "tfidf_score": round(tfidf, 4),
            "query_term_coverage": round(coverage, 4),
            "exact_term_match_count": exact_match_cnt,
            "term_frequency_sum": tf_sum,
            "document_length_norm": round(doc_len_norm, 4),
            "query_length": q_len,
            "phrase_match": phrase_match,
            "title_match": title_match,
            "fuzzy_score": round(fuzzy_score, 4)
        }

    def extract_vector(
        self, 
        query_terms: List[str], 
        doc_id: str, 
        context: Any,
        raw_query: str = "",
        fuzzy_corrections: Optional[Dict[str, str]] = None
    ) -> List[float]:
        named = self.extract_named_features(
            query_terms, 
            doc_id, 
            context, 
            raw_query=raw_query, 
            fuzzy_corrections=fuzzy_corrections
        )
        return [named[name] for name in FEATURE_NAMES]
