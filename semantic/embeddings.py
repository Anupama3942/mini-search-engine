"""
Mini Search Engine - Stage 15
Text Embedding Service & Dense Vector Encoders
"""

import math
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import config
from cache import BoundedLRUCache


def normalize_l2(vector: List[float]) -> List[float]:
    """Normalize vector to unit length (L2 norm = 1.0)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm < 1e-9:
        return [0.0] * len(vector)
    return [round(x / norm, 6) for x in vector]


class BaseEmbeddingModel(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        pass

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]


class DenseSemanticEmbeddingModel(BaseEmbeddingModel):
    """
    Deterministic subword-contextual semantic dense embedding model.
    Encodes sentences into a fixed-dimension dense vector space using
    character-ngram projections and positional weighting with L2 normalization.
    """

    def __init__(self, dimension: int = config.EMBEDDING_DIMENSION):
        self._dim = dimension
        self._name = "dense-semantic-projection-64"

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, text: str) -> List[float]:
        clean = text.lower().strip()
        if not clean:
            return [0.0] * self._dim

        words = clean.split()
        vec = [0.0] * self._dim

        # 1. Word and Character-N-gram Projection
        for pos, word in enumerate(words):
            pos_weight = 1.0 / math.sqrt(pos + 1.0)
            
            # Word-level hash embedding
            word_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            for d in range(self._dim):
                slot_hash = (word_hash ^ (d * 0x9e3779b9)) & 0xFFFFFFFF
                val = ((slot_hash % 2001) - 1000) / 1000.0
                vec[d] += val * pos_weight

            # Character 3-gram subwords for morphological and typo semantic similarity
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    trigram = word[i:i+3]
                    tri_hash = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
                    for d in range(self._dim):
                        slot_hash = (tri_hash ^ (d * 0x85ebca6b)) & 0xFFFFFFFF
                        val = ((slot_hash % 1001) - 500) / 1000.0
                        vec[d] += val * 0.4 * pos_weight

        # 2. L2 Unit Normalization
        return normalize_l2(vec)


class SentenceTransformersEmbeddingModel(BaseEmbeddingModel):
    """Wrapper for sentence-transformers library if installed."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self._name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            self.available = True
        except Exception:
            self.available = False
            self._dim = config.EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, text: str) -> List[float]:
        if not self.available:
            return [0.0] * self._dim
        vec = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.available:
            return [[0.0] * self._dim for _ in texts]
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [[float(x) for x in v] for v in vecs]


class EmbeddingService:
    """
    Singleton Embedding Service managing model lifecycle and LRU embedding cache.
    Loads the model once at application startup.
    """

    _instance: Optional["EmbeddingService"] = None

    def __init__(self):
        # Try loading sentence-transformers, or fall back to high-performance dense embedder
        st_model = SentenceTransformersEmbeddingModel()
        if st_model.available:
            self.model: BaseEmbeddingModel = st_model
        else:
            self.model: BaseEmbeddingModel = DenseSemanticEmbeddingModel()

        self.cache = BoundedLRUCache(maxsize=config.EMBEDDING_CACHE_SIZE, name="embedding_cache")

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def dimension(self) -> int:
        return self.model.dimension

    def encode(self, text: str) -> List[float]:
        """Encode a single text into a dense vector with LRU caching."""
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        vector = self.model.encode(text)
        self.cache.set(cache_key, vector)
        return vector

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts efficiently."""
        results = []
        uncached_texts = []
        uncached_indices = []

        for idx, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = self.cache.get(cache_key)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append((idx, cache_key))

        if uncached_texts:
            encoded_batch = self.model.encode_batch(uncached_texts)
            for (idx, cache_key), vec in zip(uncached_indices, encoded_batch):
                results[idx] = vec
                self.cache.set(cache_key, vec)

        return results
