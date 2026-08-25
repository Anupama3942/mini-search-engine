"""
Mini Search Engine - Stage 15
Hybrid Search: Sparse (BM25) + Dense (Semantic) Score Fusion
"""

from typing import List, Dict, Any, Optional
import config
from .embeddings import EmbeddingService
from .vector_store import NumpyVectorStore


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Safely normalize raw scores to [0.0, 1.0] range.
    Handles single-item or zero-variance cases without division by zero.
    """
    if not scores:
        return {}

    vals = list(scores.values())
    min_v = min(vals)
    max_v = max(vals)
    denom = max_v - min_v

    normalized = {}
    for doc_id, val in scores.items():
        if denom > 1e-9:
            norm_val = (val - min_v) / denom
        else:
            norm_val = 1.0 if val > 0 else 0.0
        normalized[doc_id] = round(norm_val, 6)

    return normalized


class HybridSearchEngine:
    """Combines Sparse BM25 lexical ranking and Dense Vector semantic similarity."""

    def __init__(
        self, 
        vector_store: Optional[NumpyVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.vector_store = vector_store or NumpyVectorStore()
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        if not self.vector_store.is_loaded and config.VECTOR_INDEX_PATH.exists():
            self.vector_store.load()

    def search_hybrid(
        self, 
        query: str, 
        engine, 
        alpha: float = config.HYBRID_ALPHA, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Search with candidate union and score fusion:
        hybrid_score = alpha * norm(BM25) + (1 - alpha) * norm(Semantic)
        """
        config.validate_hybrid_params(alpha)

        # 1. Sparse BM25 Candidate Retrieval (Top 20)
        bm25_results = engine.search(query, ranking_algorithm="bm25", top_k=20, log_analytics=False)
        raw_bm25_scores = {}
        if isinstance(bm25_results, list):
            for r in bm25_results:
                raw_bm25_scores[r["filename"]] = float(r["score"])

        # 2. Dense Vector Retrieval (Top 20)
        raw_semantic_scores = {}
        if self.vector_store.is_loaded:
            query_vec = self.embedding_service.encode(query)
            dense_results = self.vector_store.search(query_vec, top_k=20)
            for r in dense_results:
                raw_semantic_scores[r["filename"]] = float(r["score"])

        # 3. Candidate Union
        all_candidates = set(raw_bm25_scores.keys()) | set(raw_semantic_scores.keys())
        if not all_candidates:
            return []

        # 4. Score Normalization
        norm_bm25 = min_max_normalize(raw_bm25_scores)
        norm_semantic = min_max_normalize(raw_semantic_scores)

        # 5. Hybrid Score Fusion
        hybrid_results = []
        for doc_id in all_candidates:
            s_bm25 = norm_bm25.get(doc_id, 0.0)
            s_dense = norm_semantic.get(doc_id, 0.0)
            h_score = (alpha * s_bm25) + ((1.0 - alpha) * s_dense)

            hybrid_results.append({
                "filename": doc_id,
                "score": round(h_score, 6),
                "bm25_raw_score": raw_bm25_scores.get(doc_id, 0.0),
                "semantic_raw_score": raw_semantic_scores.get(doc_id, 0.0),
                "bm25_normalized": s_bm25,
                "semantic_normalized": s_dense,
                "alpha": alpha,
                "ranking_algorithm": "hybrid"
            })

        # Sort descending by hybrid score
        hybrid_results.sort(key=lambda x: (-x["score"], x["filename"]))
        return hybrid_results[:top_k] if top_k else hybrid_results
