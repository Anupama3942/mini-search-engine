"""
Mini Search Engine - Stage 16
Production Advanced Retrieval & Two-Stage Ranking Pipeline Benchmark
"""

import json
import csv
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


def run_production_pipeline_benchmark():
    print("=" * 80)
    print("    MINI SEARCH ENGINE - PRODUCTION RETRIEVAL PIPELINE BENCHMARK (STAGE 16)")
    print("=" * 80)

    # 1. Load Ground Truth Relevance Judgments
    with open(DEFAULT_JUDGMENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", [])
    print(f"\n[1] Loaded {len(queries)} evaluation queries.")

    service = SearchService.get_instance()

    pipelines = [
        ("BM25 (Sparse)", "bm25", None),
        ("Semantic (Dense)", "semantic", None),
        ("Hybrid (Sparse+Dense)", "hybrid", 0.5),
        ("BM25 -> LTR (Two-Stage)", "bm25_ltr", None),
        ("Hybrid -> LTR (Two-Stage)", "hybrid_ltr", 0.5),
    ]

    report_rows = []

    print("\n[2] Executing Production Retrieval Pipelines...")
    print(f"{'Pipeline':<28} | {'MAP':<8} | {'MRR':<8} | {'NDCG@5':<8} | {'P@5':<8} | {'Avg Latency':<12} | {'P95 Latency':<12}")
    print("-" * 84)

    for pipe_label, method_code, alpha in pipelines:
        ap_scores = []
        rr_scores = []
        ndcg5_scores = []
        ndcg10_scores = []
        p5_scores = []
        latencies_ms = []

        for q_item in queries:
            query_text = q_item["query"]
            ground_truth = q_item.get("relevant_documents", [])

            t_q_start = time.perf_counter()
            resp = service.search(
                query=query_text,
                method=method_code,
                top_k=10,
                alpha=alpha
            )
            lat_ms = (time.perf_counter() - t_q_start) * 1000.0
            latencies_ms.append(lat_ms)

            retrieved = [r["filename"] for r in resp.get("results", [])]

            ap = average_precision(retrieved, ground_truth)
            rr = reciprocal_rank(retrieved, ground_truth)
            nd5 = ndcg_at_k(retrieved, ground_truth, k=5)
            nd10 = ndcg_at_k(retrieved, ground_truth, k=10)
            p5 = precision_at_k(retrieved, ground_truth, k=5)

            ap_scores.append(ap)
            rr_scores.append(rr)
            ndcg5_scores.append(nd5)
            ndcg10_scores.append(nd10)
            p5_scores.append(p5)

        map_score = mean_average_precision(ap_scores)
        mrr_score = mean_reciprocal_rank(rr_scores)
        mean_nd5 = mean_ndcg_at_k(ndcg5_scores)
        mean_nd10 = mean_ndcg_at_k(ndcg10_scores)
        mean_p5 = round(sum(p5_scores) / len(p5_scores), 4)

        sorted_lat = sorted(latencies_ms)
        avg_lat = sum(sorted_lat) / len(sorted_lat)
        p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]

        row = {
            "pipeline": pipe_label,
            "method_code": method_code,
            "map": map_score,
            "mrr": mrr_score,
            "ndcg@5": mean_nd5,
            "ndcg@10": mean_nd10,
            "p@5": mean_p5,
            "avg_latency_ms": round(avg_lat, 3),
            "p95_latency_ms": round(p95_lat, 3)
        }
        report_rows.append(row)

        print(f"{pipe_label:<28} | {map_score:<8.4f} | {mrr_score:<8.4f} | {mean_nd5:<8.4f} | {mean_p5:<8.4f} | {avg_lat:<10.3f}ms | {p95_lat:<10.3f}ms")

    print("-" * 84)

    # 3. Export Reports
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEFAULT_REPORTS_DIR / "production_ranking_comparison.json"
    csv_path = DEFAULT_REPORTS_DIR / "production_ranking_comparison.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\n[3] Saved Production Benchmark Reports:\n  - JSON: {json_path}\n  - CSV:  {csv_path}\n")
    print("=" * 80)

    return report_rows


if __name__ == "__main__":
    run_production_pipeline_benchmark()
