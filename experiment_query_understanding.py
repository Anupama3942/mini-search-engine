"""
Mini Search Engine - Stage 17
Query Understanding, Synonym Expansion & Adaptive Routing Experiments
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import config
from search import SearchEngine
from services.search_service import SearchService
from evaluation.metrics import (
    average_precision,
    reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    mean_ndcg_at_k
)
from evaluation.evaluator import DEFAULT_JUDGMENTS_PATH, DEFAULT_REPORTS_DIR


def run_query_understanding_experiment():
    print("=" * 84)
    print("    MINI SEARCH ENGINE - QUERY UNDERSTANDING & ROUTING EXPERIMENT (STAGE 17)")
    print("=" * 84)

    with open(DEFAULT_JUDGMENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", [])
    print(f"\n[1] Evaluating across {len(queries)} benchmark queries.")

    service = SearchService.get_instance()

    modes = [
        ("Fixed BM25 (Baseline)", "bm25", False),
        ("Fixed Semantic", "semantic", False),
        ("Fixed Hybrid (alpha=0.5)", "hybrid", False),
        ("Query-Adaptive Routing (Auto)", "adaptive", False),
        ("Query-Adaptive + Synonyms", "adaptive", True),
    ]

    report_rows = []

    print("\n[2] Executing Evaluation Across Routing & Expansion Modes...")
    print(f"{'Strategy / Mode':<32} | {'MAP':<8} | {'MRR':<8} | {'NDCG@5':<8} | {'P@5':<8} | {'Avg Latency':<12}")
    print("-" * 86)

    for mode_label, method_code, use_synonyms in modes:
        ap_scores = []
        rr_scores = []
        ndcg5_scores = []
        p5_scores = []
        latencies_ms = []

        # Configure synonym mode
        config.SYNONYM_EXPANSION_ENABLED = use_synonyms

        for q_item in queries:
            query_text = q_item["query"]
            ground_truth = q_item.get("relevant_documents", [])

            t_start = time.perf_counter()
            resp = service.search(
                query=query_text,
                method=method_code,
                top_k=10
            )
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            latencies_ms.append(lat_ms)

            retrieved = [r["filename"] for r in resp.get("results", [])]

            ap = average_precision(retrieved, ground_truth)
            rr = reciprocal_rank(retrieved, ground_truth)
            nd5 = ndcg_at_k(retrieved, ground_truth, k=5)
            p5 = precision_at_k(retrieved, ground_truth, k=5)

            ap_scores.append(ap)
            rr_scores.append(rr)
            ndcg5_scores.append(nd5)
            p5_scores.append(p5)

        map_score = mean_average_precision(ap_scores)
        mrr_score = mean_reciprocal_rank(rr_scores)
        mean_nd5 = mean_ndcg_at_k(ndcg5_scores)
        mean_p5 = round(sum(p5_scores) / len(p5_scores), 4)
        avg_lat = round(sum(latencies_ms) / len(latencies_ms), 3)

        row = {
            "mode": mode_label,
            "method": method_code,
            "synonyms_enabled": use_synonyms,
            "map": map_score,
            "mrr": mrr_score,
            "ndcg@5": mean_nd5,
            "p@5": mean_p5,
            "avg_latency_ms": avg_lat
        }
        report_rows.append(row)

        print(f"{mode_label:<32} | {map_score:<8.4f} | {mrr_score:<8.4f} | {mean_nd5:<8.4f} | {mean_p5:<8.4f} | {avg_lat:<10.3f}ms")

    print("-" * 86)

    # Restore default configuration
    config.SYNONYM_EXPANSION_ENABLED = True

    # Save report
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = DEFAULT_REPORTS_DIR / "query_understanding_experiment.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, indent=2)

    print(f"\n[3] Saved Query Understanding Experiment Report to:\n  - JSON: {report_file}\n")
    print("=" * 84)

    return report_rows


if __name__ == "__main__":
    run_query_understanding_experiment()
