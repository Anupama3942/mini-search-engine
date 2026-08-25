"""
Mini Search Engine - Stage 15
Vector Store & Exact Vector Retrieval Index
"""

import math
import json
import time
import heapq
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import config
from .embeddings import EmbeddingService, normalize_l2


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute Cosine Similarity between two vectors:
    cos_sim(A, B) = (A · B) / (||A|| * ||B||)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    denom = norm_a * norm_b
    if denom < 1e-9:
        return 0.0

    sim = dot_product / denom
    # Clamp to [-1.0, 1.0] for numerical stability
    return round(max(min(sim, 1.0), -1.0), 6)


class VectorStore(ABC):
    """Abstract interface for dense vector indexes."""

    @abstractmethod
    def add_document(self, doc_id: str, vector: List[float]) -> None:
        pass

    @abstractmethod
    def remove_document(self, doc_id: str) -> bool:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self, index_path: Path, metadata_path: Path) -> bool:
        pass

    @abstractmethod
    def load(self, index_path: Path, metadata_path: Path) -> bool:
        pass


class NumpyVectorStore(VectorStore):
    """
    Lightweight vector storage and exact nearest neighbor retrieval engine.
    Stores dense document vectors and computes cosine similarity ranking.
    """

    def __init__(self, dimension: int = config.EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.doc_ids: List[str] = []
        self.vectors: Dict[str, List[float]] = {}
        self.model_name: str = config.EMBEDDING_MODEL_NAME
        self.is_loaded: bool = False

    def add_document(self, doc_id: str, vector: List[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} does not match index dimension {self.dimension}")
        
        normalized = normalize_l2(vector)
        if doc_id not in self.vectors:
            self.doc_ids.append(doc_id)
        self.vectors[doc_id] = normalized
        self.is_loaded = True

    def remove_document(self, doc_id: str) -> bool:
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            if doc_id in self.doc_ids:
                self.doc_ids.remove(doc_id)
            return True
        return False

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform exact cosine similarity vector retrieval against all stored documents."""
        if not self.vectors or not query_vector:
            return []

        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} does not match index dimension {self.dimension}")

        query_norm = normalize_l2(query_vector)
        scored_results = []

        for doc_id, doc_vec in self.vectors.items():
            sim = cosine_similarity(query_norm, doc_vec)
            scored_results.append({
                "filename": doc_id,
                "score": sim,
                "ranking_algorithm": "semantic"
            })

        # Top-K sorting
        if top_k and len(scored_results) > top_k:
            ranked = heapq.nsmallest(top_k, scored_results, key=lambda x: (-x["score"], x["filename"]))
        else:
            ranked = sorted(scored_results, key=lambda x: (-x["score"], x["filename"]))

        return ranked

    def build(self, documents: Dict[str, str], service: Optional[EmbeddingService] = None) -> None:
        """Batch encode and index all documents."""
        service = service or EmbeddingService.get_instance()
        self.dimension = service.dimension
        self.model_name = service.model_name
        self.vectors.clear()
        self.doc_ids.clear()

        doc_items = list(documents.items())
        texts = [content for _, content in doc_items]
        embeddings = service.encode_batch(texts)

        for (filename, _), emb in zip(doc_items, embeddings):
            self.add_document(filename, emb)

        self.is_loaded = True

    def save(
        self, 
        index_path: Path = config.VECTOR_INDEX_PATH, 
        metadata_path: Path = config.VECTOR_METADATA_PATH
    ) -> bool:
        """Serialize vector index and metadata to JSON."""
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_data = {
                "dimension": self.dimension,
                "model_name": self.model_name,
                "doc_count": len(self.vectors),
                "vectors": self.vectors
            }
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f)

            metadata = {
                "embedding_model": self.model_name,
                "embedding_dimension": self.dimension,
                "normalization_method": "l2_unit_norm",
                "document_count": len(self.vectors),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"[VectorStore Warning] Failed to save vector index: {e}")
            return False

    def load(
        self, 
        index_path: Path = config.VECTOR_INDEX_PATH, 
        metadata_path: Optional[Path] = config.VECTOR_METADATA_PATH
    ) -> bool:
        """Load precomputed vector index from JSON."""
        if not index_path.exists():
            return False
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.dimension = data.get("dimension", self.dimension)
            self.model_name = data.get("model_name", self.model_name)
            self.vectors = data.get("vectors", {})
            self.doc_ids = list(self.vectors.keys())
            self.is_loaded = len(self.vectors) > 0
            return True
        except Exception as e:
            print(f"[VectorStore Warning] Failed to load vector index: {e}")
            return False
