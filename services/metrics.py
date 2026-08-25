"""
Mini Search Engine - Stage 16
Production Metrics & Observability Registry
"""

import time
import threading
from typing import Dict, Any, List


class MetricsRegistry:
    """Thread-safe application metrics registry tracking search performance and errors."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.requests_by_method: Dict[str, int] = {}
        self.errors_by_type: Dict[str, int] = {}
        self.latencies: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self._initialized = True

    def record_request(self, method: str, latency_seconds: float, success: bool = True, error_type: str = None) -> None:
        """Record a completed search request event."""
        with self._lock:
            self.request_count += 1
            method_key = method.lower()
            self.requests_by_method[method_key] = self.requests_by_method.get(method_key, 0) + 1
            
            # Maintain sliding window of last 2000 latencies
            self.latencies.append(latency_seconds)
            if len(self.latencies) > 2000:
                self.latencies.pop(0)

            if not success:
                self.error_count += 1
                err_key = error_type or "unknown_error"
                self.errors_by_type[err_key] = self.errors_by_type.get(err_key, 0) + 1

    def record_cache_event(self, hit: bool) -> None:
        """Record cache hit or miss event."""
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def get_latency_percentiles(self) -> Dict[str, float]:
        """Compute P50, P95, P99, and Average latency in milliseconds."""
        with self._lock:
            if not self.latencies:
                return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "avg_ms": 0.0}
            sorted_l = sorted(self.latencies)
            n = len(sorted_l)
            p50 = sorted_l[int(n * 0.50)] * 1000.0
            p95 = sorted_l[min(int(n * 0.95), n - 1)] * 1000.0
            p99 = sorted_l[min(int(n * 0.99), n - 1)] * 1000.0
            avg = (sum(sorted_l) / n) * 1000.0
            return {
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
                "avg_ms": round(avg, 3)
            }

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics snapshot as a Python dictionary."""
        uptime = round(time.time() - self.start_time, 2)
        total_cache = self.cache_hits + self.cache_misses
        hit_rate = round((self.cache_hits / total_cache) * 100.0, 2) if total_cache > 0 else 0.0

        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "requests_by_method": dict(self.requests_by_method),
            "errors_by_type": dict(self.errors_by_type),
            "latency_stats": self.get_latency_percentiles(),
            "cache_stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate_pct": hit_rate
            }
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        snapshot = self.to_dict()
        lines = [
            "# HELP search_requests_total Total number of search requests processed.",
            "# TYPE search_requests_total counter",
            f"search_requests_total {snapshot['total_requests']}",
            "# HELP search_errors_total Total number of failed search requests.",
            "# TYPE search_errors_total counter",
            f"search_errors_total {snapshot['total_errors']}",
            "# HELP search_latency_p95_ms 95th percentile latency in milliseconds.",
            "# TYPE search_latency_p95_ms gauge",
            f"search_latency_p95_ms {snapshot['latency_stats']['p95_ms']}",
            "# HELP search_cache_hit_rate_pct Cache hit percentage.",
            "# TYPE search_cache_hit_rate_pct gauge",
            f"search_cache_hit_rate_pct {snapshot['cache_stats']['hit_rate_pct']}",
            "# HELP app_uptime_seconds Application uptime in seconds.",
            "# TYPE app_uptime_seconds gauge",
            f"app_uptime_seconds {snapshot['uptime_seconds']}"
        ]
        return "\n".join(lines) + "\n"


# Global singleton instance
metrics_registry = MetricsRegistry()
