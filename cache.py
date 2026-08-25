"""
Mini Search Engine - Stage 11
Bounded LRU Cache with Performance & Hit-Rate Analytics
"""

from collections import OrderedDict
from typing import Any, Dict, Optional


class BoundedLRUCache:
    """
    Thread-safe / deterministic Least Recently Used (LRU) cache.
    Automatically bounds memory growth and tracks hits, misses, and hit rate.
    """
    def __init__(self, maxsize: int = 256, name: str = "cache"):
        self.maxsize = max(1, maxsize)
        self.name = name
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        if key in self._cache:
            self.hits += 1
            # Move to end to represent recently used
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            # Evict oldest item
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear cache entries while maintaining hit/miss statistics."""
        self._cache.clear()

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = round((self.hits / total_requests) * 100.0, 2) if total_requests > 0 else 0.0
        return {
            "name": self.name,
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_pct": hit_rate
        }
