"""
Mini Search Engine - Stage 20
A/B Testing Statistical Analysis, Confidence Intervals & Hypothesis Testing
"""

import math
from typing import List, Dict, Any, Tuple, Optional


def calculate_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_variance(values: List[float], mean_val: Optional[float] = None) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean_val if mean_val is not None else calculate_mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def compare_variants(
    values_a: List[float], 
    values_b: List[float], 
    metric_name: str = "Metric",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Compute rigorous statistical comparison between Variant A (Control) and Variant B (Treatment).
    Calculates: Sample size, Means, Variance, Difference, Relative Uplift %, Standard Error,
    95% Confidence Interval, t-statistic, and statistical significance flag.
    """
    n_a = len(values_a)
    n_b = len(values_b)

    if n_a == 0 or n_b == 0:
        return {
            "metric": metric_name,
            "sample_size_a": n_a,
            "sample_size_b": n_b,
            "mean_a": 0.0,
            "mean_b": 0.0,
            "difference": 0.0,
            "relative_uplift_pct": 0.0,
            "standard_error": 0.0,
            "confidence_interval_95": [0.0, 0.0],
            "t_statistic": 0.0,
            "statistically_significant": False,
            "status": "insufficient_data"
        }

    mean_a = calculate_mean(values_a)
    mean_b = calculate_mean(values_b)
    var_a = calculate_variance(values_a, mean_a)
    var_b = calculate_variance(values_b, mean_b)

    diff = mean_b - mean_a
    uplift_pct = ((diff / mean_a) * 100.0) if mean_a != 0 else 0.0

    se = math.sqrt((var_a / n_a) + (var_b / n_b)) if (n_a > 0 and n_b > 0) else 0.0
    z_val = 1.96  # 95% Confidence
    ci_lower = round(diff - z_val * se, 4)
    ci_upper = round(diff + z_val * se, 4)

    t_stat = (diff / se) if se > 0 else 0.0
    significant = abs(t_stat) >= 1.96 and (n_a >= 10 and n_b >= 10)

    return {
        "metric": metric_name,
        "sample_size_a": n_a,
        "sample_size_b": n_b,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "difference": round(diff, 4),
        "relative_uplift_pct": round(uplift_pct, 2),
        "standard_error": round(se, 4),
        "confidence_interval_95": [ci_lower, ci_upper],
        "t_statistic": round(t_stat, 3),
        "statistically_significant": significant,
        "conclusion": "Variant B is significantly better" if (significant and diff > 0) else
                      "Variant A is significantly better" if (significant and diff < 0) else
                      "No statistically significant difference detected"
    }
