"""
Mini Search Engine - Stage 17
Synonym Dictionaries & Context-Aware Query Expansion
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional

import config


class SynonymExpander:
    """Manages synonym dictionaries and performs conservative or aggressive query expansion."""

    def __init__(self, synonyms_path: Path = config.SYNONYMS_PATH):
        self.synonyms_path = synonyms_path
        self.synonyms: Dict[str, List[str]] = {}
        self.aggressive_extensions: Dict[str, List[str]] = {}
        self.load_synonyms()

    def load_synonyms(self) -> bool:
        if not self.synonyms_path.exists():
            return False
        try:
            with open(self.synonyms_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.synonyms = {k.lower(): [v.lower() for v in vals] for k, vals in data.get("synonyms", {}).items()}
            self.aggressive_extensions = {k.lower(): [v.lower() for v in vals] for k, vals in data.get("aggressive_extensions", {}).items()}
            return True
        except Exception as e:
            print(f"[SynonymExpander Warning] Failed to load synonyms from {self.synonyms_path}: {e}")
            return False

    def get_synonyms(self, term: str, mode: str = config.QUERY_EXPANSION_MODE) -> List[str]:
        """Return matching synonyms for a single term based on the selected expansion mode."""
        if mode == "disabled" or not config.SYNONYM_EXPANSION_ENABLED:
            return []

        clean_term = term.lower().strip()
        syn_list = list(self.synonyms.get(clean_term, []))

        if mode == "aggressive":
            ext = self.aggressive_extensions.get(clean_term, [])
            for item in ext:
                if item not in syn_list:
                    syn_list.append(item)

        return syn_list[:config.MAX_EXPANSION_TERMS]

    def expand_query_terms(self, terms: List[str], mode: str = config.QUERY_EXPANSION_MODE) -> List[str]:
        """Expand a list of positive query terms with relevant synonyms."""
        expanded: Set[str] = set()
        for t in terms:
            syns = self.get_synonyms(t, mode=mode)
            for s in syns:
                if s not in terms:
                    expanded.add(s)
        return sorted(list(expanded))
