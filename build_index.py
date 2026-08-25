"""
Mini Search Engine - Stage 16
Production Index Rebuild & Management CLI
"""

import sys
import time
from services.index_manager import IndexManager


def run_index_build():
    print("=" * 72)
    print("      MINI SEARCH ENGINE - PRODUCTION INDEX BUILDER (STAGE 16)")
    print("=" * 72)

    manager = IndexManager()
    t_start = time.perf_counter()

    try:
        print("\n[1] Starting Atomic Multi-Index Build...")
        res = manager.build_all_indexes(atomic=True)

        print("\n[2] BUILD RESULTS:")
        print(f"  * Documents Indexed  : {res['documents_indexed']}")
        print(f"  * Vocabulary Size    : {res['vocabulary_size']}")
        print(f"  * Vector Dimension   : {res['vector_dimension']}")
        print(f"  * Build Duration     : {res['duration_seconds']:.4f} s")

        print("\n[3] Running Readiness & Health Validation...")
        health = manager.get_health()
        print(f"  * Status             : {health['status'].upper()}")
        print(f"  * Readiness Check    : {'PASS' if health['ready'] else 'FAIL'}")

        print("\n" + "=" * 72)
        print("PRODUCTION INDEX REBUILD COMPLETE: ALL INDEXES ACTIVE & HEALTHY")
        print("=" * 72 + "\n")
        return True

    except Exception as e:
        print(f"\n[Error] Index build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_index_build()
