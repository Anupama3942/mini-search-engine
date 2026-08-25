"""
Mini Search Engine - Stage 14
Feature Ablation Experiments for Learning-to-Rank
"""

from typing import List, Dict, Any, Tuple
from .features import FEATURE_NAMES
from .models import PointwiseLogisticRegressionModel
from .dataset import QuerySample, LTRDatasetBuilder
from evaluation.metrics import average_precision, reciprocal_rank, precision_at_k, ndcg_at_k, mean_average_precision, mean_reciprocal_rank, mean_ndcg_at_k


ABLATION_FEATURE_SETS = {
    "BM25 only": ["bm25_score"],
    "BM25 + TF-IDF": ["bm25_score", "tfidf_score"],
    "+ Coverage": ["bm25_score", "tfidf_score", "query_term_coverage"],
    "All Features": list(FEATURE_NAMES)
}


class FeatureAblationExperiment:
    """Evaluates the marginal contribution of ranking features to search relevance."""

    def __init__(self, builder: LTRDatasetBuilder):
        self.builder = builder

    def _extract_subset(
        self, 
        X: List[List[float]], 
        selected_features: List[str]
    ) -> List[List[float]]:
        indices = [FEATURE_NAMES.index(f) for f in selected_features if f in FEATURE_NAMES]
        return [[row[i] for i in indices] for row in X]

    def run_experiment(
        self, 
        train_samples: List[QuerySample], 
        test_samples: List[QuerySample]
    ) -> List[Dict[str, Any]]:
        results = []

        for set_name, feature_subset in ABLATION_FEATURE_SETS.items():
            # 1. Prepare training subset
            X_train_raw, y_train = self.builder.flatten_dataset(train_samples)
            X_train_sub = self._extract_subset(X_train_raw, feature_subset)

            # 2. Train model on subset
            model = PointwiseLogisticRegressionModel(epochs=800)
            model.fit(X_train_sub, y_train, feature_names=feature_subset)

            # 3. Evaluate on test queries
            ap_scores = []
            rr_scores = []
            ndcg5_scores = []
            p5_scores = []

            for sample in test_samples:
                X_test_sub = self._extract_subset(sample.X, feature_subset)
                preds = model.predict_proba(X_test_sub)

                # Pair doc_ids with scores and sort
                doc_scores = list(zip(sample.doc_ids, preds))
                doc_scores.sort(key=lambda x: -x[1])
                retrieved_ranked = [d for d, s in doc_scores]

                ground_truth = [sample.doc_ids[i] for i, y in enumerate(sample.y) if y == 1.0]

                ap_scores.append(average_precision(retrieved_ranked, ground_truth))
                rr_scores.append(reciprocal_rank(retrieved_ranked, ground_truth))
                ndcg5_scores.append(ndcg_at_k(retrieved_ranked, ground_truth, k=5))
                p5_scores.append(precision_at_k(retrieved_ranked, ground_truth, k=5))

            summary = {
                "feature_set": set_name,
                "features_used": feature_subset,
                "feature_count": len(feature_subset),
                "map": mean_average_precision(ap_scores),
                "mrr": mean_reciprocal_rank(rr_scores),
                "ndcg@5": mean_ndcg_at_k(ndcg5_scores),
                "p@5": round(sum(p5_scores) / len(p5_scores), 4) if p5_scores else 0.0
            }
            results.append(summary)

        return results
