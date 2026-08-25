"""
Mini Search Engine - Stage 17
Query Representation Object
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class QueryRepresentation:
    """Rich parsed and understood representation of a user search query."""
    original_query: str
    normalized_query: str
    tokens: List[str] = field(default_factory=list)
    corrected_query: Optional[str] = None
    corrections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    has_spelling_corrections: bool = False
    phrases: List[str] = field(default_factory=list)
    important_terms: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)
    intent: str = "keyword"
    suggested_strategy: str = "bm25"
    field_filters: Dict[str, str] = field(default_factory=dict)
    effective_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize representation for JSON API responses and debug views."""
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "effective_query": self.effective_query or self.normalized_query,
            "corrected_query": self.corrected_query,
            "has_spelling_corrections": self.has_spelling_corrections,
            "corrections": self.corrections,
            "detected_phrases": self.phrases,
            "important_terms": self.important_terms,
            "expanded_synonyms": self.expanded_terms,
            "intent": self.intent,
            "suggested_strategy": self.suggested_strategy,
            "field_filters": self.field_filters
        }
