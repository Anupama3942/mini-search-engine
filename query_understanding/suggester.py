"""
Mini Search Engine - Stage 17
Query Suggester & Autocomplete Service
"""

from typing import List, Set, Dict, Optional


class QuerySuggester:
    """Generates prefix-based query suggestions and autocompletions from indexed vocabulary and titles."""

    def __init__(self, vocabulary: Optional[Set[str]] = None, titles: Optional[List[str]] = None):
        self.suggestions_pool: Set[str] = set()
        if vocabulary:
            self.suggestions_pool.update(vocabulary)
        if titles:
            self.suggestions_pool.update(titles)

    def update_pool(self, vocabulary: Set[str], titles: Optional[List[str]] = None) -> None:
        self.suggestions_pool = set(vocabulary)
        if titles:
            self.suggestions_pool.update(titles)

    def suggest(self, prefix: str, limit: int = 5) -> List[str]:
        """Return autocomplete suggestions matching the query prefix."""
        clean = prefix.lower().strip()
        if not clean:
            return []

        exact_prefix_matches = []
        substring_matches = []

        for item in sorted(self.suggestions_pool):
            lower_item = item.lower()
            if lower_item.startswith(clean) and lower_item != clean:
                exact_prefix_matches.append(item)
            elif clean in lower_item and lower_item != clean:
                substring_matches.append(item)

            if len(exact_prefix_matches) >= limit:
                break

        results = (exact_prefix_matches + substring_matches)[:limit]
        return results
