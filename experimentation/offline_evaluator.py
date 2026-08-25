"""
Mini Search Engine - Stage 20
Offline A/B Experiment Evaluator
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import config
from evaluation.metrics import (
    average_precision,
    reciprocal_rank,
    ndcg_at_k,
    precision_at_k
)
from evaluation.evaluator import DEFAULT_JUDGMENTS_PATH, DEFAULT_REPORTS_DIR
from .statistics import compare_variants
from .models import Experiment


class OfflineABEvaluator:
    """Runs offline A/B evaluation between two ranking strategies on labeled relevance benchmarks."""

    def __init__(self, judgments_path: Path = DEFAULT_JUDGMENTS_PATH):
        self.judgments_path = judgments_path
        self._service = None

    @property
    def service(self):
        if self._service is None:
            from services.search_service import SearchService
            self._service = SearchService.get_instance()
        return self._service

    def run_offline_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """
        Execute offline comparison between Variant A and Variant B on ground-truth queries.
        """
        with open(self.judgments_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = data.get("queries", [])

        variant_a = experiment.variant_a_method
        variant_b = experiment.variant_b_method

        metrics_a: Dict[str, List[float]] = {
            "MAP": [], "MRR": [], "NDCG@5": [], "NDCG@10": [], "P@5": [], "Latency_ms": []
        }
        metrics_b: Dict[str, List[float]] = {
            "MAP": [], "MRR": [], "NDCG@5": [], "NDCG@10": [], "P@5": [], "Latency_ms": []
        }

        # Evaluate Variant A
        for q_item in queries:
            q_text = q_item["query"]
            rel_docs = q_item.get("relevant_documents", [])

            t0 = time.perf_counter()
            resp_a = self.service.search(query=q_text, method=variant_a, top_k=10)
            lat_a = (time.perf_counter() - t0) * 1000.0
            ret_a = [r["filename"] for r in resp_a.get("results", [])]

            metrics_a["MAP"].append(average_precision(ret_a, rel_docs))
            metrics_a["MRR"].append(reciprocal_rank(ret_a, rel_docs))
            metrics_a["NDCG@5"].append(ndcg_at_k(ret_a, rel_docs, k=5))
            metrics_a["NDCG@10"].append(ndcg_at_k(ret_a, rel_docs, k=10))
            metrics_a["P@5"].append(precision_at_k(ret_a, rel_docs, k=5))
            metrics_a["Latency_ms"].append(lat_a)

        # Evaluate Variant B
        for q_item in queries:
            q_text = q_item["query"]
            rel_docs = q_item.get("relevant_documents", [])

            t0 = time.perf_counter()
            resp_b = self.service.search(query=q_text, method=variant_b, top_k=10)
            lat_b = (time.perf_counter() - t0) * 1000.0
            ret_b = [r["filename"] for r in resp_b.get("results", [])]

            metrics_b["MAP"].append(average_precision(ret_b, rel_docs))
            metrics_b["MRR"].append(reciprocal_rank(ret_b, rel_docs))
            metrics_b["NDCG@5"].append(ndcg_at_k(ret_b, rel_docs, k=5))
            metrics_b["NDCG@10"].append(ndcg_at_k(ret_b, rel_docs, k=10))
            metrics_b["P@5"].append(precision_at_k(ret_b, rel_docs, k=5))
            metrics_b["Latency_ms"].append(lat_b)

        # Statistical Comparisons
        comparison_results = {}
        for m in ["NDCG@5", "MAP", "MRR", "NDCG@10", "P@5", "Latency_ms"]:
            comparison_results[m] = compare_variants(metrics_a[m], metrics_b[m], metric_name=m)

        report = {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "primary_metric": experiment.primary_metric,
            "query_count": len(queries),
            "variant_a": {
                "name": "A",
                "method": variant_a,
                "summary": {k: round(sum(v)/len(v), 4) for k, v in metrics_a.items()}
            },
            "variant_b": {
                "name": "B",
                "method": variant_b,
                "summary": {k: round(sum(v)/len(v), 4) for k, v in metrics_b.items()}
            },
            "statistical_analysis": comparison_results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        return report
