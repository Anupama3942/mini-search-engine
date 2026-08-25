"""
Mini Search Engine - Stage 15
Semantic Vector Retrieval Ranking Strategy
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import config
from .base import BaseRanker
from .bm25 import BM25Ranker
from semantic.embeddings import EmbeddingService
from semantic.vector_store import NumpyVectorStore, cosine_similarity


class SemanticRanker(BaseRanker):
    """Dense Vector Semantic Ranking Strategy using Cosine Similarity."""

    def __init__(
        self, 
        vector_store: Optional[NumpyVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = NumpyVectorStore()
            if config.VECTOR_INDEX_PATH.exists():
                self.vector_store.load()

        self.bm25_fallback = BM25Ranker(k1=config.BM25_K1, b=config.BM25_B)

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def is_available(self) -> bool:
        return self.vector_store.is_loaded and len(self.vector_store.vectors) > 0

    def score(self, query_terms: List[str], doc_id: str, context: Any) -> float:
        if not self.is_available:
            return self.bm25_fallback.score(query_terms, doc_id, context)

        if doc_id not in self.vector_store.vectors:
            return 0.0

        query_text = " ".join(query_terms)
        query_vec = self.embedding_service.encode(query_text)
        doc_vec = self.vector_store.vectors[doc_id]
        return cosine_similarity(query_vec, doc_vec)

    def rank(
        self, 
        query_terms: List[str], 
        candidate_docs: Any, 
        context: Any, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_available:
            return self.bm25_fallback.rank(query_terms, candidate_docs, context, top_k=top_k)

        query_text = " ".join(query_terms)
        query_vec = self.embedding_service.encode(query_text)
        
        # Dense retrieval across vector store
        all_results = self.vector_store.search(query_vec, top_k=top_k)
        return all_results

    def explain_score(self, query_terms: List[str], doc_id: str, context: Any) -> Dict[str, Any]:
        if not self.is_available:
            exp = self.bm25_fallback.explain_score(query_terms, doc_id, context)
            exp["status"] = "vector_index_unavailable_fallback_to_bm25"
            return exp

        query_text = " ".join(query_terms)
        sim = self.score(query_terms, doc_id, context)

        return {
            "algorithm": self.name,
            "doc_id": doc_id,
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.dimension,
            "cosine_similarity": sim,
            "total_score": sim
        }
