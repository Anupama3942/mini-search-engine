"""
Mini Search Engine - Stage 13
BM25 Parameter Tuner & Grid Search
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from .evaluator import SearchEvaluator, DEFAULT_JUDGMENTS_PATH
import config


DEFAULT_K1_GRID = [0.8, 1.0, 1.2, 1.5, 2.0]
DEFAULT_B_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


class BM25Tuner:
    """Performs controlled grid search to optimize BM25 hyperparameters (k1, b)."""

    def __init__(self, dataset_path: Path = DEFAULT_JUDGMENTS_PATH):
        self.dataset_path = dataset_path
        self.evaluator = SearchEvaluator(dataset_path=dataset_path)

    def tune(
        self, 
        engine, 
        k1_values: Optional[List[float]] = None, 
        b_values: Optional[List[float]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        k1_list = k1_values or DEFAULT_K1_GRID
        b_list = b_values or DEFAULT_B_GRID

        trials = []

        for k1 in k1_list:
            for b in b_list:
                # Run evaluation for this specific (k1, b) configuration
                report = self.evaluator.evaluate_engine(
                    engine, 
                    top_k=top_k, 
                    ranking_algorithm="bm25",
                    k1=k1, 
                    b=b
                )
                summary = report["summary_metrics"]
                trial_res = {
                    "k1": k1,
                    "b": b,
                    "map": summary["map"],
                    "mrr": summary["mrr"],
                    "p@1": summary["p@1"],
                    "p@5": summary["p@5"],
                    "r@5": summary["r@5"],
                    "f1_score": summary["f1_score"]
                }
                trials.append(trial_res)

        # Rank configurations: highest MAP first, then MRR
        ranked_trials = sorted(trials, key=lambda x: (-x["map"], -x["mrr"], -x["p@5"]))
        best_config = ranked_trials[0] if ranked_trials else {}

        return {
            "dataset_version": self.evaluator.dataset.get("version", "1.0"),
            "queries_evaluated": len(self.evaluator.dataset.get("queries", [])),
            "total_configurations_tested": len(trials),
            "best_configuration": best_config,
            "all_configurations": ranked_trials
        }

    def export_tuning_report(self, results: Dict[str, Any], output_path: Path) -> bool:
        """Save parameter tuning findings to JSON."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            return True
        except Exception as e:
            print(f"[BM25Tuner Warning] Failed to export tuning report: {e}")
            return False
