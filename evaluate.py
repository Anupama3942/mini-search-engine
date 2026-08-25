"""
Mini Search Engine - Stage 12, 14 & 15
Search Quality, LTR & Semantic/Hybrid Evaluation CLI Runner
"""

from pathlib import Path
import sys
from search import SearchEngine
from evaluation.evaluator import SearchEvaluator, validate_evaluation_dataset, DEFAULT_REPORTS_DIR

# Quality Gate Thresholds
MIN_MAP_THRESHOLD = 0.70
MIN_P1_THRESHOLD = 0.70
MIN_R5_THRESHOLD = 0.70
MIN_MRR_THRESHOLD = 0.75


def run_evaluation():
    print("=" * 76)
    print("    MINI SEARCH ENGINE - SEARCH QUALITY & SEMANTIC EVALUATION (STAGE 15)")
    print("=" * 76)

    # 1. Validate Dataset Integrity
    val_report = validate_evaluation_dataset()
    if not val_report["is_valid"]:
        print(f"\n[Error] Evaluation dataset is invalid ({val_report['error_count']} errors):")
        for err in val_report["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    print(f"\n[1] DATASET INTEGRITY: PASSED (Version: {val_report['version']}, Queries: {val_report['query_count']})")

    # 2. Run Search Engine Evaluation
    engine = SearchEngine()
    evaluator = SearchEvaluator()

    report = evaluator.evaluate_engine(engine, top_k=10, ranking_algorithm="bm25")
    summary = report["summary_metrics"]

    print("\n[2] OVERALL RELEVANCE & RANKING METRICS (Default BM25)")
    print("-" * 76)
    print(f"  Precision@1  (P@1)  : {summary['p@1']:.4f}   |  Recall@5   (R@5)  : {summary['r@5']:.4f}")
    print(f"  Precision@3  (P@3)  : {summary['p@3']:.4f}   |  Recall@10  (R@10) : {summary['r@10']:.4f}")
    print(f"  Precision@5  (P@5)  : {summary['p@5']:.4f}   |  F1-Score          : {summary['f1_score']:.4f}")
    print(f"  Precision@10 (P@10) : {summary['p@10']:.4f}   |  Evaluation Time   : {report['evaluation_duration_seconds']:.3f} s")
    print(f"  Mean Avg Precision (MAP) : {summary['map']:.4f}   |  NDCG@5            : {summary['ndcg@5']:.4f}")
    print(f"  Mean Recip. Rank   (MRR) : {summary['mrr']:.4f}   |  NDCG@10           : {summary['ndcg@10']:.4f}")
    print("-" * 76)

    # 3. Query Type Breakdown
    print("\n[3] QUERY TYPE BREAKDOWN")
    print(f"{'Query Type':<16} | {'Count':<6} | {'MAP':<8} | {'MRR':<8} | {'NDCG@5':<8} | {'P@1':<8} | {'R@5':<8}")
    print("-" * 76)
    for qtype, m in report["query_type_breakdown"].items():
        print(f"{qtype.capitalize():<16} | {m['query_count']:<6} | {m['map']:<8.4f} | {m['mrr']:<8.4f} | {m['ndcg@5']:<8.4f} | {m['p@1']:<8.4f} | {m['r@5']:<8.4f}")
    print("-" * 76)

    # 4. Multi-Ranking Algorithm Comparison (6 Strategies)
    ranking_comp = evaluator.compare_ranking_methods(engine)
    print("\n[4] 6-WAY RANKING COMPARISON (Frequency vs TF-IDF vs BM25 vs LTR vs Semantic vs Hybrid)")
    print(f"{'Ranking Method':<26} | {'MAP':<8} | {'MRR':<8} | {'NDCG@5':<8} | {'P@1':<8} | {'R@5':<8}")
    print("-" * 76)
    fq_m = ranking_comp["frequency_ranking"]
    tf_m = ranking_comp["tfidf_ranking"]
    bm_m = ranking_comp["bm25_ranking"]
    ltr_m = ranking_comp["ltr_ranking"]
    sem_m = ranking_comp["semantic_ranking"]
    hyb_m = ranking_comp["hybrid_ranking"]

    print(f"{'Frequency Ranking':<26} | {fq_m['map']:<8.4f} | {fq_m['mrr']:<8.4f} | {fq_m['ndcg@5']:<8.4f} | {fq_m['p@1']:<8.4f} | {fq_m['r@5']:<8.4f}")
    print(f"{'TF-IDF Ranking':<26} | {tf_m['map']:<8.4f} | {tf_m['mrr']:<8.4f} | {tf_m['ndcg@5']:<8.4f} | {tf_m['p@1']:<8.4f} | {tf_m['r@5']:<8.4f}")
    print(f"{'BM25 Ranking (k1=1.2)':<26} | {bm_m['map']:<8.4f} | {bm_m['mrr']:<8.4f} | {bm_m['ndcg@5']:<8.4f} | {bm_m['p@1']:<8.4f} | {bm_m['r@5']:<8.4f}")
    print(f"{'Learning-to-Rank (LTR)':<26} | {ltr_m['map']:<8.4f} | {ltr_m['mrr']:<8.4f} | {ltr_m['ndcg@5']:<8.4f} | {ltr_m['p@1']:<8.4f} | {ltr_m['r@5']:<8.4f}")
    print(f"{'Semantic Search (Dense)':<26} | {sem_m['map']:<8.4f} | {sem_m['mrr']:<8.4f} | {sem_m['ndcg@5']:<8.4f} | {sem_m['p@1']:<8.4f} | {sem_m['r@5']:<8.4f}")
    print(f"{'Hybrid Search (alpha=0.5)':<26} | {hyb_m['map']:<8.4f} | {hyb_m['mrr']:<8.4f} | {hyb_m['ndcg@5']:<8.4f} | {hyb_m['p@1']:<8.4f} | {hyb_m['r@5']:<8.4f}")
    print("-" * 76)

    # 5. Fuzzy Search Quality Trade-Off
    fuzzy_tradeoff = evaluator.evaluate_fuzzy_tradeoff(engine)
    if fuzzy_tradeoff:
        print("\n[5] FUZZY SEARCH TRADE-OFF (Typo Queries)")
        print(f"{'Mode':<18} | {'MAP':<8} | {'MRR':<8} | {'P@5':<8} | {'R@5':<8}")
        print("-" * 76)
        fon = fuzzy_tradeoff["fuzzy_enabled"]
        foff = fuzzy_tradeoff["fuzzy_disabled"]
        print(f"{'Fuzzy ON (Stage 9)':<18} | {fon['map']:<8.4f} | {fon['mrr']:<8.4f} | {fon['p@5']:<8.4f} | {fon['r@5']:<8.4f}")
        print(f"{'Fuzzy OFF (Exact)':<18} | {foff['map']:<8.4f} | {foff['mrr']:<8.4f} | {foff['p@5']:<8.4f} | {foff['r@5']:<8.4f}")
        print("-" * 76)

    # 6. Quality Gate Check
    passed_map = summary["map"] >= MIN_MAP_THRESHOLD
    passed_p1 = summary["p@1"] >= MIN_P1_THRESHOLD
    passed_r5 = summary["r@5"] >= MIN_R5_THRESHOLD
    passed_mrr = summary["mrr"] >= MIN_MRR_THRESHOLD
    all_passed = passed_map and passed_p1 and passed_r5 and passed_mrr

    print("\n" + "=" * 76)
    print(f"QUALITY GATE STATUS: {'[ PASS ]' if all_passed else '[ FAIL ]'}")
    print(f"  - MAP >= {MIN_MAP_THRESHOLD:.2f} : {summary['map']:.4f} ({'OK' if passed_map else 'FAILED'})")
    print(f"  - P@1 >= {MIN_P1_THRESHOLD:.2f} : {summary['p@1']:.4f} ({'OK' if passed_p1 else 'FAILED'})")
    print(f"  - R@5 >= {MIN_R5_THRESHOLD:.2f} : {summary['r@5']:.4f} ({'OK' if passed_r5 else 'FAILED'})")
    print(f"  - MRR >= {MIN_MRR_THRESHOLD:.2f} : {summary['mrr']:.4f} ({'OK' if passed_mrr else 'FAILED'})")
    print("=" * 76)

    # 7. Export Reports
    json_path = DEFAULT_REPORTS_DIR / "evaluation_report.json"
    csv_path = DEFAULT_REPORTS_DIR / "evaluation_report.csv"
    evaluator.export_report_json(report, json_path)
    evaluator.export_report_csv(report, csv_path)
    print(f"\nSaved Reports:\n  - JSON: {json_path}\n  - CSV:  {csv_path}\n")

    return report


if __name__ == "__main__":
    run_evaluation()
