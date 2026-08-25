"""
Mini Search Engine - Stage 14
Learning-to-Rank (LTR) Training & Experimentation Pipeline
"""

import json
from pathlib import Path
from typing import Dict, Any

import config
from search import SearchEngine
from ltr.dataset import LTRDatasetBuilder
from ltr.models import PointwiseLogisticRegressionModel, PairwiseRankerModel
from ltr.ablation import FeatureAblationExperiment
from ltr.features import FEATURE_NAMES, FEATURE_VERSION
from evaluation.metrics import average_precision, reciprocal_rank, precision_at_k, ndcg_at_k, mean_average_precision, mean_reciprocal_rank, mean_ndcg_at_k


def evaluate_query_samples(model, samples) -> Dict[str, float]:
    """Helper to evaluate an LTR model on a list of QuerySample objects."""
    ap_scores, rr_scores, ndcg5_scores, p1_scores, p5_scores = [], [], [], [], []

    for sample in samples:
        preds = model.predict_proba(sample.X)
        doc_scores = list(zip(sample.doc_ids, preds))
        doc_scores.sort(key=lambda x: -x[1])
        retrieved_ranked = [d for d, s in doc_scores]
        ground_truth = [sample.doc_ids[i] for i, y in enumerate(sample.y) if y == 1.0]

        ap_scores.append(average_precision(retrieved_ranked, ground_truth))
        rr_scores.append(reciprocal_rank(retrieved_ranked, ground_truth))
        ndcg5_scores.append(ndcg_at_k(retrieved_ranked, ground_truth, k=5))
        p1_scores.append(precision_at_k(retrieved_ranked, ground_truth, k=1))
        p5_scores.append(precision_at_k(retrieved_ranked, ground_truth, k=5))

    return {
        "map": mean_average_precision(ap_scores),
        "mrr": mean_reciprocal_rank(rr_scores),
        "ndcg@5": mean_ndcg_at_k(ndcg5_scores),
        "p@1": round(sum(p1_scores) / len(p1_scores), 4) if p1_scores else 0.0,
        "p@5": round(sum(p5_scores) / len(p5_scores), 4) if p5_scores else 0.0
    }


def run_ltr_training():
    print("=" * 70)
    print("      MINI SEARCH ENGINE - LEARNING-TO-RANK (LTR) TRAINING (STAGE 14)")
    print("=" * 70)

    engine = SearchEngine()
    builder = LTRDatasetBuilder()

    # 1. Dataset Construction
    print("\n[1] Extracting Feature Vectors from Ground Truth Judgments...")
    samples = builder.build_dataset(engine)
    print(f"  * Extracted {len(samples)} query groups with {len(FEATURE_NAMES)} features each.")

    # 2. Query-Level Train/Val/Test Split (Preventing Data Leakage)
    train_samples, val_samples, test_samples = builder.split_queries(
        samples, 
        train_ratio=0.70, 
        val_ratio=0.15, 
        test_ratio=0.15, 
        seed=42
    )
    print(f"\n[2] Query-Grouped Split (Zero Query Leakage):")
    print(f"  * Training Queries   : {len(train_samples)} queries")
    print(f"  * Validation Queries : {len(val_samples)} queries")
    print(f"  * Test Queries       : {len(test_samples)} queries")

    X_train, y_train = builder.flatten_dataset(train_samples)
    print(f"  * Total Training Samples (Query-Doc Pairs): {len(X_train)} (Positive: {int(sum(y_train))}, Negative: {len(y_train) - int(sum(y_train))})")

    # 3. Hyperparameter Tuning over Regularization C
    print("\n[3] Hyperparameter Tuning on Validation Set:")
    best_c = 1.0
    best_val_map = -1.0
    tuning_records = []

    for c_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model_trial = PointwiseLogisticRegressionModel(epochs=1000, regularization_c=c_val)
        model_trial.fit(X_train, y_train, feature_names=FEATURE_NAMES)
        val_metrics = evaluate_query_samples(model_trial, val_samples)
        tuning_records.append({"c": c_val, "val_map": val_metrics["map"], "val_mrr": val_metrics["mrr"], "val_ndcg5": val_metrics["ndcg@5"]})
        print(f"  - Regularization C={c_val:<6} -> Val MAP={val_metrics['map']:.4f} | Val MRR={val_metrics['mrr']:.4f} | Val NDCG@5={val_metrics['ndcg@5']:.4f}")

        if val_metrics["map"] > best_val_map:
            best_val_map = val_metrics["map"]
            best_c = c_val

    print(f"  * Best Regularization Parameter Selected: C = {best_c}")

    # 4. Final Pointwise Model Training with Best Parameters
    print("\n[4] Training Final Pointwise Logistic Regression Model...")
    final_model = PointwiseLogisticRegressionModel(epochs=1000, regularization_c=best_c)
    final_model.fit(X_train, y_train, feature_names=FEATURE_NAMES)

    train_res = evaluate_query_samples(final_model, train_samples)
    val_res = evaluate_query_samples(final_model, val_samples)
    test_res = evaluate_query_samples(final_model, test_samples)

    print(f"\n[5] MODEL EVALUATION (Train vs. Val vs. Test):")
    print(f"{'Split':<12} | {'MAP':<8} | {'MRR':<8} | {'NDCG@5':<8} | {'P@1':<8} | {'P@5':<8}")
    print("-" * 62)
    print(f"{'Train':<12} | {train_res['map']:<8.4f} | {train_res['mrr']:<8.4f} | {train_res['ndcg@5']:<8.4f} | {train_res['p@1']:<8.4f} | {train_res['p@5']:<8.4f}")
    print(f"{'Validation':<12} | {val_res['map']:<8.4f} | {val_res['mrr']:<8.4f} | {val_res['ndcg@5']:<8.4f} | {val_res['p@1']:<8.4f} | {val_res['p@5']:<8.4f}")
    print(f"{'Test':<12} | {test_res['map']:<8.4f} | {test_res['mrr']:<8.4f} | {test_res['ndcg@5']:<8.4f} | {test_res['p@1']:<8.4f} | {test_res['p@5']:<8.4f}")
    print("-" * 62)

    # 6. Feature Importance / Learned Weights
    print(f"\n[6] LEARNED FEATURE WEIGHTS (Logistic Regression Coefficients):")
    weights = final_model.get_feature_importances()
    for feat_name, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        print(f"  * {feat_name:<24} : {w:>8.4f}")

    # 7. Pairwise LTR Experiment
    print(f"\n[7] PAIRWISE LTR EXPERIMENT (Preference Difference Learning):")
    pairwise_diffs = builder.generate_pairwise_differences(train_samples)
    pairwise_model = PairwiseRankerModel(epochs=600)
    pairwise_model.fit_pairs(pairwise_diffs, feature_names=FEATURE_NAMES)
    pairwise_test_res = evaluate_query_samples(pairwise_model, test_samples)
    print(f"  * Generated {len(pairwise_diffs)} Preference Training Pairs (x_rel - x_nonrel)")
    print(f"  * Pairwise Test Metrics -> MAP: {pairwise_test_res['map']:.4f} | MRR: {pairwise_test_res['mrr']:.4f} | NDCG@5: {pairwise_test_res['ndcg@5']:.4f}")

    # 8. Feature Ablation Experiments
    print(f"\n[8] FEATURE ABLATION EXPERIMENTS (Evaluating Feature Subsets):")
    ablation = FeatureAblationExperiment(builder)
    ablation_results = ablation.run_experiment(train_samples, test_samples)
    print(f"{'Feature Set':<20} | {'Features':<8} | {'MAP':<8} | {'MRR':<8} | {'NDCG@5':<8} | {'P@5':<8}")
    print("-" * 68)
    for ab in ablation_results:
        print(f"{ab['feature_set']:<20} | {ab['feature_count']:<8} | {ab['map']:<8.4f} | {ab['mrr']:<8.4f} | {ab['ndcg@5']:<8.4f} | {ab['p@5']:<8.4f}")
    print("-" * 68)

    # 9. Save Trained Model and Metadata
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save(config.LTR_MODEL_PATH, config.LTR_METADATA_PATH)
    print(f"\n[9] Saved Trained LTR Model to: {config.LTR_MODEL_PATH}")
    print(f"    Saved Metadata to:           {config.LTR_METADATA_PATH}")

    # Save Experiment Summary Report
    exp_report = {
        "model_type": "Pointwise Logistic Regression",
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "best_regularization_c": best_c,
        "feature_weights": weights,
        "evaluation_metrics": {
            "train": train_res,
            "validation": val_res,
            "test": test_res,
            "pairwise_test": pairwise_test_res
        },
        "feature_ablation": ablation_results
    }
    exp_path = config.MODELS_DIR / "ltr_experiment_report.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(exp_report, f, indent=2)
    print(f"    Saved Experiment Report to:  {exp_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_ltr_training()
