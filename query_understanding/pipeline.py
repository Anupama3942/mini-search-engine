"""
Mini Search Engine - Stage 17
Unified Query Understanding Pipeline
"""

import re
from typing import Dict, Any, List, Optional, Set

import config
from .representation import QueryRepresentation
from .normalizer import QueryNormalizer
from .spelling import SpellChecker
from .synonyms import SynonymExpander
from .intent import IntentClassifier
from .suggester import QuerySuggester


class QueryUnderstandingPipeline:
    """
    Executes sequential query understanding: Normalization -> Spell Correction ->
    Phrase Detection -> Important Term Extraction -> Synonym Expansion -> Intent Classification.
    """

    def __init__(
        self,
        vocabulary: Optional[Set[str]] = None,
        term_frequencies: Optional[Dict[str, int]] = None,
        document_titles: Optional[List[str]] = None
    ):
        self.normalizer = QueryNormalizer()
        self.spell_checker = SpellChecker(vocabulary=vocabulary, term_frequencies=term_frequencies)
        self.synonym_expander = SynonymExpander()
        self.intent_classifier = IntentClassifier()
        self.suggester = QuerySuggester(vocabulary=vocabulary, titles=document_titles)

    def update_context(
        self, 
        vocabulary: Set[str], 
        term_frequencies: Optional[Dict[str, int]] = None,
        document_titles: Optional[List[str]] = None
    ) -> None:
        self.spell_checker.update_vocabulary(vocabulary, term_frequencies)
        self.suggester.update_pool(vocabulary, document_titles)

    def analyze(self, raw_query: str) -> QueryRepresentation:
        """Analyze a raw query and return a rich QueryRepresentation."""
        if not raw_query or not raw_query.strip():
            return QueryRepresentation(original_query="", normalized_query="", intent="empty")

        # 1. Query Normalization
        normalized = self.normalizer.normalize(raw_query)

        # 2. Phrase Detection (Quoted substrings)
        phrases = []
        for m in re.finditer(r'"([^"]+)"', normalized):
            p_text = m.group(1).strip()
            if p_text:
                phrases.append(p_text)

        # 3. Field Filter Detection (e.g. title:python)
        field_filters = {}
        cleaned_for_tokens = normalized
        for m in re.finditer(r'(\b\w+):([^\s()]+)', normalized):
            field_name, field_val = m.group(1).lower(), m.group(2).lower()
            field_filters[field_name] = field_val
            cleaned_for_tokens = cleaned_for_tokens.replace(m.group(0), field_val)

        # 4. Tokenization
        raw_words = [w.strip() for w in re.findall(r'[^\s()"]+', cleaned_for_tokens) if w.strip()]

        # 5. Spelling Correction with Confidence
        corrections: Dict[str, Dict[str, Any]] = {}
        corrected_words = []
        has_corrections = False

        for word in raw_words:
            if word.upper() in ('AND', 'OR', 'NOT'):
                corrected_words.append(word.upper())
                continue

            if config.SPELL_CORRECTION_ENABLED:
                suggested, conf, was_fixed = self.spell_checker.check_word(word)
                if was_fixed:
                    corrections[word] = {"suggestion": suggested, "confidence": conf}
                    corrected_words.append(suggested)
                    has_corrections = True
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        corrected_query = None
        if has_corrections:
            # Reconstruct corrected query preserving quotes and Boolean operators
            corrected_query = normalized
            for orig_w, info in corrections.items():
                pattern = re.compile(rf"\b{re.escape(orig_w)}\b", re.IGNORECASE)
                corrected_query = pattern.sub(info["suggestion"], corrected_query)

        # 6. Important Term Extraction
        positive_terms = [w.lower() for w in corrected_words if w.upper() not in ('AND', 'OR', 'NOT')]
        important_terms = [w for w in positive_terms if len(w) > 2] or positive_terms

        # 7. Synonym Expansion (Safe from negations)
        expanded_synonyms = []
        if config.SYNONYM_EXPANSION_ENABLED and config.QUERY_EXPANSION_MODE != "disabled":
            expanded_synonyms = self.synonym_expander.expand_query_terms(
                important_terms, 
                mode=config.QUERY_EXPANSION_MODE
            )

        # 8. Intent Classification & Strategy Recommendation
        intent, suggested_strategy = self.intent_classifier.classify(normalized, raw_words)

        effective_q = corrected_query if (has_corrections and config.SPELL_CORRECTION_ENABLED) else normalized

        return QueryRepresentation(
            original_query=raw_query,
            normalized_query=normalized,
            tokens=raw_words,
            corrected_query=corrected_query,
            corrections=corrections,
            has_spelling_corrections=has_corrections,
            phrases=phrases,
            important_terms=important_terms,
            expanded_terms=expanded_synonyms,
            intent=intent,
            suggested_strategy=suggested_strategy,
            field_filters=field_filters,
            effective_query=effective_q
        )
