"""
Mini Search Engine - Stage 17
Query Intent Classification & Strategy Routing
"""

import re
from typing import Tuple, List, Dict, Any


INFORMATIONAL_TRIGGERS = {
    "how", "what", "why", "where", "when", "who", "which",
    "guide", "tutorial", "learn", "learning", "understand",
    "best", "difference", "concepts", "overview", "introduction",
    "overview", "explain", "principles", "patterns"
}


class IntentClassifier:
    """Classifies user query intent and recommends the optimal retrieval strategy."""

    @staticmethod
    def classify(query: str, tokens: List[str]) -> Tuple[str, str]:
        """
        Returns (intent, suggested_strategy).
        Intent categories: "boolean", "phrase", "navigational", "informational", "keyword", "mixed"
        Strategy options: "bm25", "semantic", "hybrid", "ltr"
        """
        clean_q = query.strip()
        lower_q = clean_q.lower()
        word_tokens = [t.lower() for t in tokens if isinstance(t, str) and t.isalnum()]

        # 1. Boolean Query Intent
        if any(op in tokens for op in ('AND', 'OR', 'NOT', '(', ')')) or re.search(r"\b(AND|OR|NOT)\b", clean_q):
            return "boolean", "bm25"

        # 2. Exact Phrase Intent
        if '"' in clean_q:
            return "phrase", "bm25"

        # 3. Navigational / Field-Specific Intent
        if ":" in clean_q:
            return "navigational", "bm25"

        # 4. Informational / Natural Language Intent
        if any(w in INFORMATIONAL_TRIGGERS for w in word_tokens) or len(word_tokens) >= 5:
            return "informational", "semantic"

        # 5. Short Technical Keyword Intent
        if len(word_tokens) <= 2:
            return "keyword", "bm25"

        # 6. Mixed Conceptual Query Intent
        return "mixed", "hybrid"
