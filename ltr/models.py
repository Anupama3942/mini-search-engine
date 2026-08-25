"""
Mini Search Engine - Stage 14
Learning-to-Rank Models: Pointwise Logistic Regression & Pairwise Ranker
"""

import math
import json
import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import config
from .features import FEATURE_NAMES, FEATURE_VERSION, FeatureScaler


def sigmoid(z: float) -> float:
    """Numerically safe sigmoid activation function."""
    z_clamped = max(min(z, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-z_clamped))


class BaseLTRModel(ABC):
    """Abstract interface for Learning-to-Rank algorithms."""

    @abstractmethod
    def fit(self, X: List[List[float]], y: List[float], feature_names: List[str]) -> "BaseLTRModel":
        pass

    @abstractmethod
    def predict_proba(self, X: List[List[float]]) -> List[float]:
        pass

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[float]:
        pass

    @abstractmethod
    def save(self, model_path: Path, metadata_path: Path) -> bool:
        pass

    @abstractmethod
    def load(self, model_path: Path) -> bool:
        pass


class PointwiseLogisticRegressionModel(BaseLTRModel):
    """
    Pointwise LTR Baseline: Logistic Regression classifier with L2 regularization
    predicting probability of relevance P(Relevant | Features).
    """

    def __init__(
        self,
        learning_rate: float = config.LTR_LEARNING_RATE,
        epochs: int = config.LTR_EPOCHS,
        regularization_c: float = config.LTR_DEFAULT_REGULARIZATION_C
    ):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.c = float(regularization_c)
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.feature_version: str = FEATURE_VERSION
        self.scaler = FeatureScaler()
        self.is_trained: bool = False

    def fit(
        self, 
        X: List[List[float]], 
        y: List[float], 
        feature_names: Optional[List[str]] = None
    ) -> "PointwiseLogisticRegressionModel":
        if not X or not y:
            raise ValueError("Training dataset X and y must not be empty.")

        if feature_names:
            self.feature_names = list(feature_names)
        num_samples = len(X)
        num_features = len(X[0])

        # 1. Fit scaler on training features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # 2. Initialize weights and bias
        random.seed(42)
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(num_features)]
        self.bias = 0.0

        # Regularization lambda = 1.0 / C
        l2_lambda = (1.0 / self.c) if self.c > 0 else 0.0

        # 3. Gradient Descent Optimization
        for _ in range(self.epochs):
            # Compute predictions
            preds = []
            for row in X_scaled:
                z = self.bias + sum(w * x for w, x in zip(self.weights, row))
                preds.append(sigmoid(z))

            # Compute gradients
            errors = [p - target for p, target in zip(preds, y)]
            
            grad_bias = sum(errors) / num_samples
            grad_weights = [0.0] * num_features
            for j in range(num_features):
                feature_grad = sum(errors[i] * X_scaled[i][j] for i in range(num_samples)) / num_samples
                # L2 penalty
                grad_weights[j] = feature_grad + (l2_lambda * self.weights[j] / num_samples)

            # Update parameters
            self.bias -= self.learning_rate * grad_bias
            for j in range(num_features):
                self.weights[j] -= self.learning_rate * grad_weights[j]

        self.is_trained = True
        return self

    def predict_proba_vector(self, feature_vector: List[float]) -> float:
        """Predict relevance probability for a single raw feature vector."""
        if not self.is_trained:
            return 0.0
        scaled_vec = self.scaler.transform_vector(feature_vector)
        z = self.bias + sum(w * x for w, x in zip(self.weights, scaled_vec))
        return sigmoid(z)

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [self.predict_proba_vector(row) for row in X]

    def predict(self, X: List[List[float]]) -> List[float]:
        """Binary classification threshold at 0.5."""
        return [1.0 if p >= 0.5 else 0.0 for p in self.predict_proba(X)]

    def get_feature_importances(self) -> Dict[str, float]:
        """Return learned model weights per feature."""
        if not self.weights:
            return {}
        return {
            name: round(self.weights[i], 6)
            for i, name in enumerate(self.feature_names)
            if i < len(self.weights)
        }

    def explain_prediction(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Break down score into individual feature linear contributions."""
        scaled_vec = self.scaler.transform_vector(feature_vector)
        contributions = {}
        total_z = self.bias

        for i, name in enumerate(self.feature_names):
            if i < len(self.weights) and i < len(scaled_vec):
                val = scaled_vec[i]
                w = self.weights[i]
                contrib = w * val
                total_z += contrib
                contributions[name] = {
                    "raw_value": feature_vector[i],
                    "scaled_value": val,
                    "weight": round(w, 4),
                    "contribution": round(contrib, 4)
                }

        prob = sigmoid(total_z)
        return {
            "model_type": "Pointwise Logistic Regression",
            "bias": round(self.bias, 4),
            "linear_sum_z": round(total_z, 4),
            "predicted_probability": round(prob, 6),
            "feature_contributions": contributions
        }

    def save(self, model_path: Path = config.LTR_MODEL_PATH, metadata_path: Path = config.LTR_METADATA_PATH) -> bool:
        """Save model parameters and metadata to JSON."""
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                "model_type": "pointwise_logistic_regression",
                "feature_version": self.feature_version,
                "feature_names": self.feature_names,
                "weights": self.weights,
                "bias": self.bias,
                "scaler": self.scaler.to_dict(),
                "hyperparameters": {
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "regularization_c": self.c
                }
            }
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(model_data, f, indent=2)

            metadata = {
                "model_type": "Pointwise Logistic Regression",
                "feature_version": self.feature_version,
                "feature_count": len(self.feature_names),
                "feature_names": self.feature_names,
                "feature_importances": self.get_feature_importances(),
                "status": "trained",
                "saved_at": model_path.name
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"[PointwiseLogisticRegression Warning] Failed to save model: {e}")
            return False

    def load(self, model_path: Path = config.LTR_MODEL_PATH) -> bool:
        """Load model parameters from JSON and validate feature version."""
        if not model_path.exists():
            return False
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            ver = data.get("feature_version")
            if ver != FEATURE_VERSION:
                print(f"[PointwiseLogisticRegression Warning] Feature version mismatch: model={ver}, system={FEATURE_VERSION}")
                return False

            self.feature_version = ver
            self.feature_names = data.get("feature_names", FEATURE_NAMES)
            self.weights = data.get("weights", [])
            self.bias = data.get("bias", 0.0)
            self.scaler = FeatureScaler.from_dict(data.get("scaler", {}))
            
            hp = data.get("hyperparameters", {})
            self.learning_rate = hp.get("learning_rate", config.LTR_LEARNING_RATE)
            self.epochs = hp.get("epochs", config.LTR_EPOCHS)
            self.c = hp.get("regularization_c", config.LTR_DEFAULT_REGULARIZATION_C)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"[PointwiseLogisticRegression Warning] Failed to load model: {e}")
            return False


class PairwiseRankerModel(BaseLTRModel):
    """
    Pairwise LTR: Trains on preference feature differences (x_rel - x_nonrel -> 1).
    Ranks individual documents via linear projection: score(D) = w^T * x.
    """

    def __init__(self, learning_rate: float = 0.05, epochs: int = 500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: List[float] = []
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.feature_version: str = FEATURE_VERSION
        self.scaler = FeatureScaler()
        self.is_trained: bool = False

    def fit_pairs(
        self, 
        X_diffs: List[List[float]], 
        feature_names: Optional[List[str]] = None
    ) -> "PairwiseRankerModel":
        if not X_diffs:
            return self
        if feature_names:
            self.feature_names = list(feature_names)

        num_samples = len(X_diffs)
        num_features = len(X_diffs[0])
        
        self.scaler.fit(X_diffs)
        X_scaled = self.scaler.transform(X_diffs)

        random.seed(42)
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(num_features)]

        # Target label is 1.0 (preferred document > non-preferred)
        for _ in range(self.epochs):
            errors = []
            for row in X_scaled:
                z = sum(w * x for w, x in zip(self.weights, row))
                p = sigmoid(z)
                errors.append(p - 1.0) # error against label 1

            for j in range(num_features):
                grad = sum(errors[i] * X_scaled[i][j] for i in range(num_samples)) / num_samples
                self.weights[j] -= self.learning_rate * grad

        self.is_trained = True
        return self

    def fit(self, X: List[List[float]], y: List[float], feature_names: Optional[List[str]] = None) -> "PairwiseRankerModel":
        # Handled via fit_pairs
        return self

    def predict_score(self, feature_vector: List[float]) -> float:
        """Linear ranking score for a single document."""
        if not self.is_trained:
            return 0.0
        scaled = self.scaler.transform_vector(feature_vector)
        return sum(w * x for w, x in zip(self.weights, scaled))

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [sigmoid(self.predict_score(row)) for row in X]

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self.predict_score(row) for row in X]

    def save(self, model_path: Path, metadata_path: Optional[Path] = None) -> bool:
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "model_type": "pairwise_ranker",
                "feature_version": self.feature_version,
                "feature_names": self.feature_names,
                "weights": self.weights,
                "scaler": self.scaler.to_dict()
            }
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

    def load(self, model_path: Path) -> bool:
        if not model_path.exists():
            return False
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.weights = data.get("weights", [])
            self.feature_names = data.get("feature_names", FEATURE_NAMES)
            self.scaler = FeatureScaler.from_dict(data.get("scaler", {}))
            self.is_trained = True
            return True
        except Exception:
            return False
