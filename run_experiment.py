"""
Mini Search Engine - Stage 20
Offline A/B Experiment Runner & Statistical Comparison CLI
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

from experimentation import ExperimentRegistry, OfflineABEvaluator
from evaluation.evaluator import DEFAULT_REPORTS_DIR


def run_ab_experiments():
    print("=" * 88)
    print("       MINI SEARCH ENGINE - OFFLINE A/B EXPERIMENTATION PLATFORM (STAGE 20)")
    print("=" * 88)

    registry = ExperimentRegistry.get_instance()
    evaluator = OfflineABEvaluator()
    experiments = registry.experiments.values()

    all_reports = []

    for exp in experiments:
        print(f"\n[+] Running A/B Experiment: '{exp.name}' (ID: {exp.id})")
        print(f"    * Control (Variant A)   : {exp.variant_a_method.upper()}")
        print(f"    * Treatment (Variant B) : {exp.variant_b_method.upper()}")
        print(f"    * Primary Metric        : {exp.primary_metric}")

        res = evaluator.run_offline_experiment(exp)
        all_reports.append(res)

        print("\n    STATISTICAL RESULTS (N = %d queries):" % res["query_count"])
        print(f"    {'Metric':<12} | {'Variant A':<10} | {'Variant B':<10} | {'Diff':<10} | {'Uplift %':<10} | {'95% CI':<18} | {'Significant?'}")
        print("    " + "-" * 82)

        for m_name in ["NDCG@5", "MAP", "MRR", "NDCG@10", "P@5", "Latency_ms"]:
            stat = res["statistical_analysis"][m_name]
            ci_str = f"[{stat['confidence_interval_95'][0]:.4f}, {stat['confidence_interval_95'][1]:.4f}]"
            sig_str = "YES (*)" if stat["statistically_significant"] else "No"
            print(f"    {m_name:<12} | {stat['mean_a']:<10.4f} | {stat['mean_b']:<10.4f} | {stat['difference']:<+10.4f} | {stat['relative_uplift_pct']:<+9.2f}% | {ci_str:<18} | {sig_str}")

        print("    " + "-" * 82)
        print(f"    Conclusion: {res['statistical_analysis'][exp.primary_metric]['conclusion']}")

    # Save aggregated reports
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEFAULT_REPORTS_DIR / "experiment_ab_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print("\n" + "=" * 88)
    print(f"Saved A/B Experiment Reports to:\n  - JSON: {json_path}")
    print("=" * 88 + "\n")

    return all_reports


if __name__ == "__main__":
    run_ab_experiments()
