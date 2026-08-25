"""
Mini Search Engine - Stage 10
Performance Monitoring & Metrics Utilities
"""

import time
import math
import tracemalloc
from typing import List, Dict, Any

# Start tracemalloc if not already started
if not tracemalloc.is_tracing():
    tracemalloc.start()


class PerformanceTimer:
    """
    Lightweight stopwatch timer for measuring execution durations with time.perf_counter().
    Can be used as a context manager or manually with start()/stop().
    """
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        if self.start_time is not None:
            self.end_time = time.perf_counter()
            self.duration = self.end_time - self.start_time
        return self.duration


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive descriptive statistics and latency percentiles:
      - Min, Max, Average
      - Median (P50), P95, P99
    Uses the nearest-rank method on sorted values.
    """
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def get_percentile(p: float) -> float:
        # p is a fraction between 0.0 and 1.0
        if n == 1:
            return sorted_vals[0]
        rank = math.ceil(p * n) - 1
        rank = max(0, min(rank, n - 1))
        return sorted_vals[rank]

    avg_val = sum(sorted_vals) / n

    return {
        "count": n,
        "min": round(sorted_vals[0], 6),
        "max": round(sorted_vals[-1], 6),
        "avg": round(avg_val, 6),
        "p50": round(get_percentile(0.50), 6),
        "p95": round(get_percentile(0.95), 6),
        "p99": round(get_percentile(0.99), 6)
    }


def get_memory_usage() -> Dict[str, Any]:
    """
    Measure Python memory allocations using standard library tracemalloc.
    Returns current and peak memory allocations formatted in KB and MB.
    """
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()

    return {
        "current_bytes": current_bytes,
        "peak_bytes": peak_bytes,
        "current_kb": round(current_bytes / 1024, 2),
        "peak_kb": round(peak_bytes / 1024, 2),
        "current_mb": round(current_bytes / (1024 * 1024), 3),
        "peak_mb": round(peak_bytes / (1024 * 1024), 3),
        "note": "Represents Python process heap memory allocated via tracemalloc."
    }
