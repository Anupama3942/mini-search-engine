"""
Mini Search Engine - Stage 13
Search Quality Evaluator & Relevance Testing Framework
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from .metrics import (
    precision,
    recall,
    f1_score,
    precision_at_k,
    recall_at_k,
    average_precision,
    reciprocal_rank,
    mean_average_precision,
    mean_reciprocal_rank,
    calculate_confusion_matrix
)
import config

DEFAULT_JUDGMENTS_PATH = Path(__file__).parent / "relevance_judgments.json"
DEFAULT_REPORTS_DIR = Path(__file__).parent / "reports"


def validate_evaluation_dataset(dataset_path: Path = DEFAULT_JUDGMENTS_PATH) -> Dict[str, Any]:
    """Validate ground truth evaluation dataset integrity."""
    errors = []
    if not dataset_path.exists():
        return {
            "is_valid": False,
            "error_count": 1,
            "errors": [f"Evaluation dataset not found at {dataset_path}"],
            "query_count": 0
        }

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "is_valid": False,
            "error_count": 1,
            "errors": [f"Failed to parse JSON dataset: {e}"],
            "query_count": 0
        }

    queries = data.get("queries", [])
    if not queries:
        errors.append("Dataset contains 0 queries.")

    seen_query_ids = set()
    for idx, item in enumerate(queries, start=1):
        qid = item.get("query_id")
        if not qid:
            errors.append(f"Query at index {idx} missing query_id.")
        elif qid in seen_query_ids:
            errors.append(f"Duplicate query_id '{qid}' at index {idx}.")
        else:
            seen_query_ids.add(qid)

        query_text = item.get("query")
        if not query_text or not str(query_text).strip():
            errors.append(f"Query '{qid}' has empty query string.")

        relevant_docs = item.get("relevant_documents")
        if relevant_docs is None or not isinstance(relevant_docs, list):
            errors.append(f"Query '{qid}' has invalid relevant_documents format.")
        elif len(relevant_docs) != len(set(relevant_docs)):
            errors.append(f"Query '{qid}' contains duplicate relevant document IDs.")

    return {
        "is_valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "query_count": len(queries),
        "version": data.get("version", "1.0")
    }


class SearchEvaluator:
    """Evaluates SearchEngine relevance and ranking quality against ground truth."""
    
    def __init__(self, dataset_path: Path = DEFAULT_JUDGMENTS_PATH):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dict[str, Any]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_engine(
        self, 
        engine, 
        top_k: int = 10,
        ranking_algorithm: str = config.DEFAULT_RANKING_ALGORITHM,
        k1: Optional[float] = None,
        b: Optional[float] = None
    ) -> Dict[str, Any]:
        """Run all evaluation queries and compute overall and per-query IR metrics."""
        queries = self.dataset.get("queries", [])
        all_documents = list(engine.documents.keys())
        
        per_query_results = []
        ap_scores = []
        rr_scores = []
        p1_scores = []
        p3_scores = []
        p5_scores = []
        p10_scores = []
        r5_scores = []
        r10_scores = []
        overall_precisions = []
        overall_recalls = []

        type_metrics = {}

        t_start = time.perf_counter()

        for q_item in queries:
            qid = q_item["query_id"]
            query_str = q_item["query"]
            qtype = q_item.get("query_type", "normal")
            ground_truth = q_item.get("relevant_documents", [])

            # Run search with specified ranking strategy
            search_results = engine.search(
                query_str, 
                log_analytics=False, 
                top_k=top_k,
                ranking_algorithm=ranking_algorithm,
                k1=k1,
                b=b
            )
            retrieved_docs = [r["filename"] for r in search_results] if not isinstance(search_results, dict) else []

            # Compute individual metrics
            p = precision(retrieved_docs, ground_truth)
            r = recall(retrieved_docs, ground_truth)
            f1 = f1_score(p, r)
            p1 = precision_at_k(retrieved_docs, ground_truth, 1)
            p3 = precision_at_k(retrieved_docs, ground_truth, 3)
            p5 = precision_at_k(retrieved_docs, ground_truth, 5)
            p10 = precision_at_k(retrieved_docs, ground_truth, 10)
            r5 = recall_at_k(retrieved_docs, ground_truth, 5)
            r10 = recall_at_k(retrieved_docs, ground_truth, 10)
            ap = average_precision(retrieved_docs, ground_truth)
            rr = reciprocal_rank(retrieved_docs, ground_truth)
            cm = calculate_confusion_matrix(retrieved_docs, ground_truth, all_documents)

            # Error Analysis breakdown
            relevant_set = set(ground_truth)
            retrieved_set = set(retrieved_docs)
            false_positives = [d for d in retrieved_docs if d not in relevant_set]
            false_negatives = [d for d in ground_truth if d not in retrieved_set]

            q_eval = {
                "query_id": qid,
                "query": query_str,
                "query_type": qtype,
                "relevant_count": len(ground_truth),
                "retrieved_count": len(retrieved_docs),
                "precision": p,
                "recall": r,
                "f1_score": f1,
                "p@1": p1,
                "p@3": p3,
                "p@5": p5,
                "p@10": p10,
                "r@5": r5,
                "r@10": r10,
                "average_precision": ap,
                "reciprocal_rank": rr,
                "confusion_matrix": cm,
                "retrieved_documents": retrieved_docs,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            }

            per_query_results.append(q_eval)
            ap_scores.append(ap)
            rr_scores.append(rr)
            p1_scores.append(p1)
            p3_scores.append(p3)
            p5_scores.append(p5)
            p10_scores.append(p10)
            r5_scores.append(r5)
            r10_scores.append(r10)
            overall_precisions.append(p)
            overall_recalls.append(r)

            # Accumulate per-type metrics
            if qtype not in type_metrics:
                type_metrics[qtype] = {"ap": [], "rr": [], "p1": [], "p5": [], "r5": [], "count": 0}
            type_metrics[qtype]["ap"].append(ap)
            type_metrics[qtype]["rr"].append(rr)
            type_metrics[qtype]["p1"].append(p1)
            type_metrics[qtype]["p5"].append(p5)
            type_metrics[qtype]["r5"].append(r5)
            type_metrics[qtype]["count"] += 1

        eval_duration = round(time.perf_counter() - t_start, 4)

        # Macro averages
        mean_p = round(sum(overall_precisions) / len(overall_precisions), 4) if overall_precisions else 0.0
        mean_r = round(sum(overall_recalls) / len(overall_recalls), 4) if overall_recalls else 0.0
        mean_f1 = f1_score(mean_p, mean_r)
        mean_p1 = round(sum(p1_scores) / len(p1_scores), 4) if p1_scores else 0.0
        mean_p3 = round(sum(p3_scores) / len(p3_scores), 4) if p3_scores else 0.0
        mean_p5 = round(sum(p5_scores) / len(p5_scores), 4) if p5_scores else 0.0
        mean_p10 = round(sum(p10_scores) / len(p10_scores), 4) if p10_scores else 0.0
        mean_r5 = round(sum(r5_scores) / len(r5_scores), 4) if r5_scores else 0.0
        mean_r10 = round(sum(r10_scores) / len(r10_scores), 4) if r10_scores else 0.0
        map_score = mean_average_precision(ap_scores)
        mrr_score = mean_reciprocal_rank(rr_scores)

        # Query Type Summaries
        query_type_summary = {}
        for qtype, m in type_metrics.items():
            query_type_summary[qtype] = {
                "query_count": m["count"],
                "map": round(sum(m["ap"]) / len(m["ap"]), 4) if m["ap"] else 0.0,
                "mrr": round(sum(m["rr"]) / len(m["rr"]), 4) if m["rr"] else 0.0,
                "p@1": round(sum(m["p1"]) / len(m["p1"]), 4) if m["p1"] else 0.0,
                "p@5": round(sum(m["p5"]) / len(m["p5"]), 4) if m["p5"] else 0.0,
                "r@5": round(sum(m["r5"]) / len(m["r5"]), 4) if m["r5"] else 0.0,
            }

        # Identify Best and Worst Queries
        sorted_by_ap = sorted(per_query_results, key=lambda x: (-x["average_precision"], -x["p@5"]))
        best_queries = sorted_by_ap[:5]
        worst_queries = sorted(per_query_results, key=lambda x: (x["average_precision"], x["p@5"]))[:5]

        report = {
            "evaluation_version": self.dataset.get("version", "1.0"),
            "ranking_algorithm": ranking_algorithm,
            "bm25_params": {"k1": k1 or config.BM25_K1, "b": b or config.BM25_B} if ranking_algorithm == "bm25" else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_duration_seconds": eval_duration,
            "queries_evaluated": len(queries),
            "summary_metrics": {
                "precision": mean_p,
                "recall": mean_r,
                "f1_score": mean_f1,
                "p@1": mean_p1,
                "p@3": mean_p3,
                "p@5": mean_p5,
                "p@10": mean_p10,
                "r@5": mean_r5,
                "r@10": mean_r10,
                "map": map_score,
                "mrr": mrr_score
            },
            "query_type_breakdown": query_type_summary,
            "best_performing_queries": [
                {"id": q["query_id"], "query": q["query"], "ap": q["average_precision"], "rr": q["reciprocal_rank"]}
                for q in best_queries
            ],
            "worst_performing_queries": [
                {"id": q["query_id"], "query": q["query"], "ap": q["average_precision"], "rr": q["reciprocal_rank"], "false_positives": len(q["false_positives"]), "false_negatives": len(q["false_negatives"])}
                for q in worst_queries
            ],
            "per_query_results": per_query_results
        }

        return report

    def compare_ranking_methods(self, engine, top_k: int = 10) -> Dict[str, Any]:
        """Compare all 3 ranking algorithms: BM25, TF-IDF, and Frequency."""
        bm25_rep = self.evaluate_engine(engine, top_k=top_k, ranking_algorithm="bm25")
        tfidf_rep = self.evaluate_engine(engine, top_k=top_k, ranking_algorithm="tfidf")
        freq_rep = self.evaluate_engine(engine, top_k=top_k, ranking_algorithm="frequency")

        return {
            "bm25_ranking": {
                "map": bm25_rep["summary_metrics"]["map"],
                "mrr": bm25_rep["summary_metrics"]["mrr"],
                "p@1": bm25_rep["summary_metrics"]["p@1"],
                "p@5": bm25_rep["summary_metrics"]["p@5"],
                "r@5": bm25_rep["summary_metrics"]["r@5"],
            },
            "tfidf_ranking": {
                "map": tfidf_rep["summary_metrics"]["map"],
                "mrr": tfidf_rep["summary_metrics"]["mrr"],
                "p@1": tfidf_rep["summary_metrics"]["p@1"],
                "p@5": tfidf_rep["summary_metrics"]["p@5"],
                "r@5": tfidf_rep["summary_metrics"]["r@5"],
            },
            "frequency_ranking": {
                "map": freq_rep["summary_metrics"]["map"],
                "mrr": freq_rep["summary_metrics"]["mrr"],
                "p@1": freq_rep["summary_metrics"]["p@1"],
                "p@5": freq_rep["summary_metrics"]["p@5"],
                "r@5": freq_rep["summary_metrics"]["r@5"],
            }
        }

    def evaluate_fuzzy_tradeoff(self, engine) -> Dict[str, Any]:
        """Evaluate quality impact of Fuzzy Search on typo queries (Fuzzy ON vs OFF)."""
        fuzzy_queries = [q for q in self.dataset.get("queries", []) if q.get("query_type") == "fuzzy"]
        if not fuzzy_queries:
            return {}

        # 1. Fuzzy ON (standard)
        on_ap, on_rr, on_p1, on_p5, on_r5 = [], [], [], [], []
        for q in fuzzy_queries:
            gt = q.get("relevant_documents", [])
            res = engine.search(q["query"], log_analytics=False)
            ret = [r["filename"] for r in res] if isinstance(res, list) else []
            on_ap.append(average_precision(ret, gt))
            on_rr.append(reciprocal_rank(ret, gt))
            on_p1.append(precision_at_k(ret, gt, 1))
            on_p5.append(precision_at_k(ret, gt, 5))
            on_r5.append(recall_at_k(ret, gt, 5))

        # 2. Fuzzy OFF (exact lookup without correction)
        off_ap, off_rr, off_p1, off_p5, off_r5 = [], [], [], [], []
        for q in fuzzy_queries:
            gt = q.get("relevant_documents", [])
            tokens = q["query"].lower().split()
            ret_docs = set()
            for t in tokens:
                ret_docs |= engine.inverted_index.get(t, set())
            ret = list(ret_docs)
            off_ap.append(average_precision(ret, gt))
            off_rr.append(reciprocal_rank(ret, gt))
            off_p1.append(precision_at_k(ret, gt, 1))
            off_p5.append(precision_at_k(ret, gt, 5))
            off_r5.append(recall_at_k(ret, gt, 5))

        return {
            "fuzzy_enabled": {
                "map": mean_average_precision(on_ap),
                "mrr": mean_reciprocal_rank(on_rr),
                "p@1": round(sum(on_p1) / len(on_p1), 4),
                "p@5": round(sum(on_p5) / len(on_p5), 4),
                "r@5": round(sum(on_r5) / len(on_r5), 4)
            },
            "fuzzy_disabled": {
                "map": mean_average_precision(off_ap),
                "mrr": mean_reciprocal_rank(off_rr),
                "p@1": round(sum(off_p1) / len(off_p1), 4),
                "p@5": round(sum(off_p5) / len(off_p5), 4),
                "r@5": round(sum(off_r5) / len(off_r5), 4)
            }
        }

    def export_report_json(self, report: Dict[str, Any], output_path: Path) -> bool:
        """Export evaluation report to JSON."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return True
        except Exception as e:
            print(f"[SearchEvaluator Warning] Failed to export JSON report: {e}")
            return False

    def export_report_csv(self, report: Dict[str, Any], output_path: Path) -> bool:
        """Export per-query evaluation breakdown to CSV."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            per_query = report.get("per_query_results", [])
            if not per_query:
                return False

            fieldnames = [
                "query_id", "query", "query_type", "relevant_count", "retrieved_count",
                "precision", "recall", "f1_score", "p@1", "p@3", "p@5", "p@10",
                "r@5", "r@10", "average_precision", "reciprocal_rank"
            ]

            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_query:
                    filtered_row = {k: row[k] for k in fieldnames if k in row}
                    writer.writerow(filtered_row)
            return True
        except Exception as e:
            print(f"[SearchEvaluator Warning] Failed to export CSV report: {e}")
            return False
