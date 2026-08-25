"""
Mini Search Engine - Stage 15
Hybrid Search Alpha Parameter Experiment Tool
"""

import json
from pathlib import Path
from search import SearchEngine
from evaluation.evaluator import SearchEvaluator, DEFAULT_REPORTS_DIR
import config


def run_hybrid_experiment():
    print("=" * 70)
    print("      MINI SEARCH ENGINE - HYBRID SEARCH ALPHA EXPERIMENT (STAGE 15)")
    print("=" * 70)

    engine = SearchEngine()
    evaluator = SearchEvaluator()

    alpha_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = []

    print("\nRunning Grid Search over alpha in [0.0 (Pure Semantic) -> 1.0 (Pure BM25)]...")
    print(f"{'Alpha (BM25 Weight)':<20} | {'MAP':<8} | {'MRR':<8} | {'NDCG@5':<8} | {'P@1':<8} | {'R@5':<8}")
    print("-" * 70)

    for alpha in alpha_grid:
        report = evaluator.evaluate_engine(
            engine, 
            top_k=10, 
            ranking_algorithm="hybrid",
            alpha=alpha
        )
        summary = report["summary_metrics"]
        res = {
            "alpha": alpha,
            "mode": "Pure Semantic" if alpha == 0.0 else ("Pure BM25" if alpha == 1.0 else f"Hybrid (alpha={alpha})"),
            "map": summary["map"],
            "mrr": summary["mrr"],
            "ndcg@5": summary["ndcg@5"],
            "p@1": summary["p@1"],
            "r@5": summary["r@5"]
        }
        results.append(res)
        print(f"{res['mode']:<20} | {res['map']:<8.4f} | {res['mrr']:<8.4f} | {res['ndcg@5']:<8.4f} | {res['p@1']:<8.4f} | {res['r@5']:<8.4f}")

    print("-" * 70)

    # Save results to reports directory
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = DEFAULT_REPORTS_DIR / "hybrid_alpha_experiment.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Hybrid Experiment Report to: {report_file}\n")
    print("=" * 70)


if __name__ == "__main__":
    run_hybrid_experiment()
