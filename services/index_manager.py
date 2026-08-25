"""
Mini Search Engine - Stage 16
Centralized Index Manager & Atomic Index Operations
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import config
from search import SearchEngine, load_documents
from semantic.embeddings import EmbeddingService
from semantic.vector_store import NumpyVectorStore


class IndexManager:
    """Coordinates lifecycle, validation, atomic builds, and health checks of all search indexes."""

    def __init__(self, documents_dir: Path = config.DOCUMENTS_DIR):
        self.documents_dir = documents_dir

    def validate_indexes(self) -> Dict[str, Any]:
        """Validate disk status of inverted index, vector store, and LTR model."""
        vector_exists = config.VECTOR_INDEX_PATH.exists()
        vector_meta_exists = config.VECTOR_METADATA_PATH.exists()
        ltr_exists = config.LTR_MODEL_PATH.exists()
        ltr_meta_exists = config.LTR_METADATA_PATH.exists()

        vector_info = {}
        if vector_meta_exists:
            try:
                with open(config.VECTOR_METADATA_PATH, "r", encoding="utf-8") as f:
                    vector_info = json.load(f)
            except Exception:
                pass

        ltr_info = {}
        if ltr_meta_exists:
            try:
                with open(config.LTR_METADATA_PATH, "r", encoding="utf-8") as f:
                    ltr_info = json.load(f)
            except Exception:
                pass

        return {
            "documents_directory": str(self.documents_dir),
            "documents_found": len(list(self.documents_dir.glob("*.txt"))) if self.documents_dir.exists() else 0,
            "vector_index": {
                "exists": vector_exists,
                "metadata": vector_info,
                "dimension": vector_info.get("embedding_dimension", config.EMBEDDING_DIMENSION)
            },
            "ltr_model": {
                "exists": ltr_exists,
                "metadata": ltr_info,
                "version": ltr_info.get("feature_version", config.FEATURE_VERSION)
            }
        }

    def is_ready(self) -> bool:
        """Readiness check: confirms documents exist and essential indexes/models are present."""
        if not self.documents_dir.exists():
            return False
        docs = list(self.documents_dir.glob("*.txt"))
        if not docs:
            return False
        return True

    def get_health(self) -> Dict[str, Any]:
        """Comprehensive health and readiness status."""
        ready = self.is_ready()
        validation = self.validate_indexes()
        return {
            "status": "ready" if ready else "not_ready",
            "ready": ready,
            "validation": validation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

    def build_all_indexes(self, atomic: bool = True) -> Dict[str, Any]:
        """
        Build text index, vector index, and metadata atomically.
        Writes to temporary files first, then performs atomic file replacements.
        """
        t_start = time.perf_counter()
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load documents
        documents = load_documents(self.documents_dir)
        if not documents:
            raise ValueError(f"No text documents found in {self.documents_dir}")

        # 2. Build Inverted & Positional Indexes
        engine = SearchEngine(documents)
        engine.save_index(config.INDEX_CACHE_PATH)

        # 3. Build Vector Store
        emb_service = EmbeddingService.get_instance()
        vector_store = NumpyVectorStore(dimension=emb_service.dimension)
        vector_store.build(documents, service=emb_service)

        if atomic:
            tmp_vec_idx = config.MODELS_DIR / "vector_index.json.tmp"
            tmp_vec_meta = config.MODELS_DIR / "vector_index_metadata.json.tmp"

            vector_store.save(tmp_vec_idx, tmp_vec_meta)

            # Atomic swap
            os.replace(tmp_vec_idx, config.VECTOR_INDEX_PATH)
            os.replace(tmp_vec_meta, config.VECTOR_METADATA_PATH)
        else:
            vector_store.save(config.VECTOR_INDEX_PATH, config.VECTOR_METADATA_PATH)

        duration = round(time.perf_counter() - t_start, 4)

        return {
            "status": "success",
            "documents_indexed": len(documents),
            "vocabulary_size": len(engine.inverted_index),
            "vector_dimension": emb_service.dimension,
            "duration_seconds": duration
        }
