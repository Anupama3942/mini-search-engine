"""
Mini Search Engine - Stage 15
Offline Vector Index Construction Tool
"""

import time
from pathlib import Path
import config
from search import load_documents
from semantic.embeddings import EmbeddingService
from semantic.vector_store import NumpyVectorStore


def build_vector_index():
    print("=" * 70)
    print("     MINI SEARCH ENGINE - VECTOR INDEX BUILDER (STAGE 15)")
    print("=" * 70)

    t_start = time.perf_counter()

    # 1. Load documents
    print("\n[1] Loading Corpus Documents...")
    documents = load_documents(config.DOCUMENTS_DIR)
    if not documents:
        print("[Error] No documents found in documents directory.")
        return False
    print(f"  * Found {len(documents)} documents to encode.")

    # 2. Initialize Embedding Service
    print("\n[2] Initializing Embedding Model...")
    service = EmbeddingService.get_instance()
    print(f"  * Embedding Model     : {service.model_name}")
    print(f"  * Embedding Dimension : {service.dimension}")

    # 3. Batch Encode Documents
    print("\n[3] Generating Dense Vector Embeddings...")
    vector_store = NumpyVectorStore(dimension=service.dimension)
    vector_store.build(documents, service=service)

    # 4. Save Vector Store and Metadata
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save(config.VECTOR_INDEX_PATH, config.VECTOR_METADATA_PATH)

    build_time = time.perf_counter() - t_start
    print(f"\n[4] VECTOR INDEX SUCCESSFULLY BUILT:")
    print(f"  * Total Documents Indexed : {len(vector_store.vectors)}")
    print(f"  * Vector Index File       : {config.VECTOR_INDEX_PATH}")
    print(f"  * Vector Metadata File     : {config.VECTOR_METADATA_PATH}")
    print(f"  * Total Build Duration    : {build_time*1000:.2f} ms ({len(documents)/build_time:.1f} docs/sec)")
    print("=" * 70)

    return True


if __name__ == "__main__":
    build_vector_index()
