"""
Mini Search Engine - Stage 20
A/B Experiment Registry & Configuration Manager
"""

from typing import Dict, List, Optional, Tuple, Any
import config
from .models import Experiment


class ExperimentRegistry:
    """Central registry of active and configured search A/B experiments."""

    _instance = None

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self._register_default_experiments()

    @classmethod
    def get_instance(cls) -> "ExperimentRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_default_experiments(self) -> None:
        """Register default production ranking experiments."""
        # 1. BM25 vs Hybrid Search
        self.register(Experiment(
            id="bm25_vs_hybrid",
            name="BM25 vs Hybrid Search Fusion",
            description="Comparing sparse BM25 probabilistic ranking against dense+sparse Hybrid retrieval.",
            enabled=True,
            traffic_percentage=100.0,
            variant_a_method="bm25",
            variant_b_method="hybrid",
            split_ratio=0.50,
            primary_metric="NDCG@5"
        ))

        # 2. BM25 vs Dense Semantic Search
        self.register(Experiment(
            id="bm25_vs_semantic",
            name="BM25 vs Dense Semantic Search",
            description="Comparing exact lexical BM25 against dense semantic vector embeddings.",
            enabled=True,
            traffic_percentage=100.0,
            variant_a_method="bm25",
            variant_b_method="semantic",
            split_ratio=0.50,
            primary_metric="MAP"
        ))

        # 3. BM25->LTR vs Hybrid->LTR (Two-Stage Pipeline)
        self.register(Experiment(
            id="bm25_ltr_vs_hybrid_ltr",
            name="Two-Stage Pipeline Reranking",
            description="Comparing BM25 candidate retrieval with Hybrid candidate retrieval prior to LTR reranking.",
            enabled=True,
            traffic_percentage=100.0,
            variant_a_method="bm25_ltr",
            variant_b_method="hybrid_ltr",
            split_ratio=0.50,
            primary_metric="NDCG@5"
        ))

        # 4. Query-Adaptive Routing vs Baseline BM25
        self.register(Experiment(
            id="adaptive_vs_bm25",
            name="Query-Adaptive Routing vs Fixed BM25",
            description="Comparing NLP intent-driven query routing against fixed BM25 ranking.",
            enabled=True,
            traffic_percentage=100.0,
            variant_a_method="bm25",
            variant_b_method="adaptive",
            split_ratio=0.50,
            primary_metric="NDCG@5"
        ))

    def register(self, experiment: Experiment) -> None:
        self.experiments[experiment.id] = experiment

    def get(self, experiment_id: str) -> Optional[Experiment]:
        return self.experiments.get(experiment_id)

    def assign(self, experiment_id: str, entity_id: str) -> Tuple[Optional[str], str]:
        """
        Assign an entity to an experiment variant.
        Returns Tuple[variant_name, ranking_method].
        Falls back safely to (None, DEFAULT_RANKING_ALGORITHM) if disabled or invalid.
        """
        if not config.EXPERIMENTS_ENABLED:
            return None, config.DEFAULT_RANKING_ALGORITHM

        exp = self.get(experiment_id)
        if not exp or not exp.enabled:
            return None, config.DEFAULT_RANKING_ALGORITHM

        assignment = exp.assign_variant(entity_id)
        if assignment is None:
            return None, exp.variant_a_method

        return assignment[0], assignment[1]

    def list_experiments(self) -> List[Dict[str, Any]]:
        return [exp.to_dict() for exp in self.experiments.values()]
