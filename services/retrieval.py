"""
Mini Search Engine - Stage 16
Retrieval Layer Abstractions & Candidate Retrievers
"""

from abc import ABC, abstractmethod
from typing import List, Set, Any, Optional
import config
from semantic.embeddings import EmbeddingService
from semantic.vector_store import NumpyVectorStore


class BaseRetriever(ABC):
    """Abstract interface for candidate retrieval engines."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = config.CANDIDATE_POOL_SIZE) -> List[str]:
        """Retrieve top-K candidate document IDs matching the query."""
        pass


class BM25Retriever(BaseRetriever):
    """Sparse Lexical Candidate Retriever using BM25 index."""

    def __init__(self, search_engine):
        self.engine = search_engine

    def retrieve(self, query: str, top_k: int = config.CANDIDATE_POOL_SIZE) -> List[str]:
        results = self.engine.search(
            query=query, 
            top_k=top_k, 
            ranking_algorithm="bm25", 
            log_analytics=False
        )
        if isinstance(results, list):
            return [r["filename"] for r in results]
        return []


class SemanticRetriever(BaseRetriever):
    """Dense Vector Semantic Candidate Retriever."""

    def __init__(self, vector_store: Optional[NumpyVectorStore] = None, service: Optional[EmbeddingService] = None):
        self.service = service or EmbeddingService.get_instance()
        self.vector_store = vector_store or NumpyVectorStore()
        if not self.vector_store.is_loaded and config.VECTOR_INDEX_PATH.exists():
            self.vector_store.load()

    def retrieve(self, query: str, top_k: int = config.CANDIDATE_POOL_SIZE) -> List[str]:
        if not self.vector_store.is_loaded:
            return []
        query_vec = self.service.encode(query)
        results = self.vector_store.search(query_vec, top_k=top_k)
        return [r["filename"] for r in results]


class HybridRetriever(BaseRetriever):
    """Hybrid (Sparse + Dense) Candidate Retriever taking the union of top-K results."""

    def __init__(self, search_engine, vector_store: Optional[NumpyVectorStore] = None):
        self.sparse_retriever = BM25Retriever(search_engine)
        self.dense_retriever = SemanticRetriever(vector_store)

    def retrieve(self, query: str, top_k: int = config.CANDIDATE_POOL_SIZE) -> List[str]:
        sparse_candidates = self.sparse_retriever.retrieve(query, top_k=top_k)
        dense_candidates = self.dense_retriever.retrieve(query, top_k=top_k)
        
        # Preserve order while building union
        seen: Set[str] = set()
        union: List[str] = []
        for doc_id in sparse_candidates + dense_candidates:
            if doc_id not in seen:
                seen.add(doc_id)
                union.append(doc_id)

        return union[:top_k] if top_k else union
