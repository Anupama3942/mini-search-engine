"""
Mini Search Engine - Stage 17
Spelling Correction with Confidence Scoring
"""

import math
from typing import Dict, Tuple, Optional, Set, Any
import config


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance using Dynamic Programming."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]


class SpellChecker:
    """
    Intelligent query-level spell corrector using corpus vocabulary and term frequencies
    to assign probabilistic confidence to corrections.
    """

    def __init__(self, vocabulary: Optional[Set[str]] = None, term_frequencies: Optional[Dict[str, int]] = None):
        self.vocabulary = set(vocabulary or [])
        self.term_frequencies = dict(term_frequencies or {})

    def update_vocabulary(self, vocabulary: Set[str], term_frequencies: Optional[Dict[str, int]] = None) -> None:
        self.vocabulary = set(vocabulary)
        if term_frequencies:
            self.term_frequencies = dict(term_frequencies)

    def check_word(self, word: str, max_distance: int = 2) -> Tuple[str, float, bool]:
        """
        Check a single word and return (best_candidate, confidence, was_corrected).
        """
        clean_word = word.lower().strip()
        if not clean_word or len(clean_word) <= 2:
            return clean_word, 1.0, False

        # If already in vocabulary, perfect match
        if clean_word in self.vocabulary:
            return clean_word, 1.0, False

        best_candidate = clean_word
        best_distance = max_distance + 1
        best_freq = 0
        best_conf = 0.0

        for vocab_term in self.vocabulary:
            # Length pre-filter optimization
            if abs(len(vocab_term) - len(clean_word)) > max_distance:
                continue

            dist = levenshtein_distance(clean_word, vocab_term)
            if dist <= max_distance:
                freq = self.term_frequencies.get(vocab_term, 1)
                
                # Confidence formulation
                sim = 1.0 - (dist / max(len(clean_word), len(vocab_term)))
                freq_boost = min(1.0, math.log(1 + freq) / math.log(1 + 5))
                conf = round(sim * (0.8 + 0.2 * freq_boost), 4)

                if dist < best_distance or (dist == best_distance and freq > best_freq):
                    best_distance = dist
                    best_candidate = vocab_term
                    best_freq = freq
                    best_conf = conf

        was_corrected = (best_candidate != clean_word and best_conf >= config.SPELL_CORRECTION_THRESHOLD)
        return (best_candidate if was_corrected else clean_word), best_conf, was_corrected
