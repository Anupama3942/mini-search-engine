"""
Mini Search Engine - Stage 20
A/B Experiment Data Models & Deterministic Variant Assignment
"""

import hashlib
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class Experiment:
    """Represents a controlled A/B search ranking experiment."""
    id: str
    name: str
    description: str
    enabled: bool = True
    traffic_percentage: float = 100.0   # % of traffic enrolled (0..100)
    variant_a_method: str = "bm25"     # Control variant
    variant_b_method: str = "hybrid"   # Treatment variant
    split_ratio: float = 0.50          # 0.5 for 50/50, 0.9 for 90/10
    primary_metric: str = "NDCG@5"
    secondary_metrics: List[str] = field(default_factory=lambda: ["MAP", "MRR", "Latency", "CTR", "ZeroResultRate"])

    def assign_variant(self, entity_id: str) -> Optional[Tuple[str, str]]:
        """
        Deterministically assign an entity (session_id, user_id, or query_id) to a variant.
        Returns Tuple[variant_name, ranking_method] e.g. ("A", "bm25") or None if excluded.
        """
        if not self.enabled:
            return None

        # 1. Deterministic hashing using SHA-256 for uniform hash distribution
        raw_key = f"{self.id}:{entity_id}".encode("utf-8")
        hash_val = int(hashlib.sha256(raw_key).hexdigest(), 16)
        bucket = hash_val % 100  # 0 to 99

        # 2. Check traffic gating
        if bucket >= self.traffic_percentage:
            return None  # Excluded from experiment

        # 3. Deterministic split
        split_threshold = int(self.split_ratio * 100)
        if bucket < split_threshold:
            return "A", self.variant_a_method
        else:
            return "B", self.variant_b_method

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "traffic_percentage": self.traffic_percentage,
            "variant_a": {"name": "A", "method": self.variant_a_method},
            "variant_b": {"name": "B", "method": self.variant_b_method},
            "split_ratio": f"{int(self.split_ratio*100)}/{int((1-self.split_ratio)*100)}",
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics
        }
