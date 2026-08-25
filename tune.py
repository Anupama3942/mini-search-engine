"""
Mini Search Engine - Stage 13
BM25 Parameter Tuning CLI Tool
"""

from pathlib import Path
from search import SearchEngine
from evaluation.tuner import BM25Tuner
from evaluation.evaluator import DEFAULT_REPORTS_DIR
import config


def run_tuning():
    print("=" * 70)
    print("        MINI SEARCH ENGINE - BM25 PARAMETER TUNING (STAGE 13)")
    print("=" * 70)

    engine = SearchEngine()
    tuner = BM25Tuner()

    print("\nRunning Grid Search over k1 in [0.8, 1.0, 1.2, 1.5, 2.0] and b in [0.0, 0.25, 0.5, 0.75, 1.0]...")
    results = tuner.tune(engine)

    print(f"\n[1] TESTED {results['total_configurations_tested']} CONFIGURATIONS ON DATASET v{results['dataset_version']}")
    print(f"{'k1':<6} | {'b':<6} | {'MAP':<8} | {'MRR':<8} | {'P@1':<8} | {'R@5':<8} | {'F1':<8}")
    print("-" * 70)

    for cfg in results["all_configurations"]:
        print(f"{cfg['k1']:<6.2f} | {cfg['b']:<6.2f} | {cfg['map']:<8.4f} | {cfg['mrr']:<8.4f} | {cfg['p@1']:<8.4f} | {cfg['r@5']:<8.4f} | {cfg['f1_score']:<8.4f}")
    print("-" * 70)

    best = results["best_configuration"]
    print(f"\n[2] RECOMMENDED CONFIGURATION:")
    print(f"  * k1  : {best['k1']}")
    print(f"  * b   : {best['b']}")
    print(f"  * MAP : {best['map']:.4f}")
    print(f"  * MRR : {best['mrr']:.4f}")
    print(f"  * P@1 : {best['p@1']:.4f}")
    print(f"  * R@5 : {best['r@5']:.4f}")
    print("=" * 70)

    # Export report
    report_path = DEFAULT_REPORTS_DIR / "bm25_tuning_report.json"
    tuner.export_tuning_report(results, report_path)
    print(f"\nSaved Tuning Report to: {report_path}\n")

    return results


if __name__ == "__main__":
    run_tuning()
