"""
Mini Search Engine - Stage 17
Query Normalization & Character Preprocessing
"""

import re
import unicodedata


class QueryNormalizer:
    """Normalizes raw query strings while strictly preserving Boolean and Phrase syntax."""

    @staticmethod
    def normalize(query: str) -> str:
        if not query:
            return ""

        # 1. Unicode NFKC Normalization (standardize composite characters)
        normalized = unicodedata.normalize("NFKC", str(query))

        # 2. Trim and collapse repeated whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # 3. Clean up non-operator noisy punctuation (e.g. trailing exclamation marks, questions)
        # Preserve: quotes '"', parentheses '(', ')', field colon ':', and alphanumeric characters
        cleaned_chars = []
        in_quotes = False

        for char in normalized:
            if char == '"':
                in_quotes = not in_quotes
                cleaned_chars.append(char)
            elif in_quotes:
                cleaned_chars.append(char)
            elif char in ('(', ')', ':', '-', '_') or char.isalnum() or char.isspace():
                cleaned_chars.append(char)
            elif char in ('!', '?', ';', ',', '.', '~', '$', '%', '^', '*', '+', '=', '|', '<', '>', '/'):
                # Replace noisy punctuation with a single space outside quotes
                cleaned_chars.append(' ')

        result = "".join(cleaned_chars)
        # Collapse spaces again
        result = re.sub(r"\s+", " ", result).strip()

        return result
