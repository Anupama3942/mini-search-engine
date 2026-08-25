"""
Mini Search Engine - Stage 14
Learning-to-Rank (LTR) Strategy with BM25 Fallback
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import config
from .base import BaseRanker
from .bm25 import BM25Ranker
from ltr.features import FeatureExtractor, FEATURE_NAMES, FEATURE_VERSION
from ltr.models import PointwiseLogisticRegressionModel


class LTRRanker(BaseRanker):
    """
    Learning-to-Rank strategy utilizing Pointwise Logistic Regression
    with graceful fallback to BM25 if the model is missing or incompatible.
    """

    def __init__(self, model_path: Path = config.LTR_MODEL_PATH):
        self.model_path = model_path
        self.extractor = FeatureExtractor()
        self.bm25_fallback = BM25Ranker(k1=config.BM25_K1, b=config.BM25_B)
        self.model = PointwiseLogisticRegressionModel()
        self.is_ready = self._initialize_model()

    def _initialize_model(self) -> bool:
        if not self.model_path.exists():
            return False
        success = self.model.load(self.model_path)
        return success and self.model.is_trained

    @property
    def name(self) -> str:
        return "ltr"

    def score(self, query_terms: List[str], doc_id: str, context: Any) -> float:
        if not self.is_ready:
            # Safe Fallback to BM25
            return self.bm25_fallback.score(query_terms, doc_id, context)

        # Extract features and predict probability
        vec = self.extractor.extract_vector(query_terms, doc_id, context)
        prob = self.model.predict_proba_vector(vec)
        return prob

    def explain_score(self, query_terms: List[str], doc_id: str, context: Any) -> Dict[str, Any]:
        if not self.is_ready:
            explanation = self.bm25_fallback.explain_score(query_terms, doc_id, context)
            explanation["status"] = "model_unavailable_fallback_to_bm25"
            return explanation

        named_features = self.extractor.extract_named_features(query_terms, doc_id, context)
        vec = [named_features[k] for k in FEATURE_NAMES]
        model_explanation = self.model.explain_prediction(vec)

        return {
            "algorithm": self.name,
            "doc_id": doc_id,
            "feature_version": FEATURE_VERSION,
            "total_score": model_explanation["predicted_probability"],
            "model_details": model_explanation
        }
