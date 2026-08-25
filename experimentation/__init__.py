"""
Mini Search Engine - Stage 20 Search Experimentation Package
"""

from .models import Experiment
from .registry import ExperimentRegistry
from .statistics import compare_variants, calculate_mean, calculate_variance
from .offline_evaluator import OfflineABEvaluator

__all__ = [
    "Experiment",
    "ExperimentRegistry",
    "compare_variants",
    "calculate_mean",
    "calculate_variance",
    "OfflineABEvaluator"
]
