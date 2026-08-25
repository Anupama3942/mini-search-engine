import unittest
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from query_understanding import (
    QueryNormalizer,
    SpellChecker,
    SynonymExpander,
    IntentClassifier,
    QuerySuggester,
    QueryUnderstandingPipeline,
    QueryRepresentation,
    levenshtein_distance
)
from services.search_service import SearchService
from app import app
import config


class TestQueryUnderstanding(unittest.TestCase):

    def setUp(self):
        self.pipeline = QueryUnderstandingPipeline(
            vocabulary={"python", "programming", "database", "networking", "web", "development"},
            term_frequencies={"python": 10, "programming": 8, "database": 5},
            document_titles=["Python", "Web", "Database", "Networking"]
        )

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("python", "python"), 0)
        self.assertEqual(levenshtein_distance("pythn", "python"), 1)

    def test_query_normalizer(self):
        norm = QueryNormalizer()
        raw = "   Python   Programming!!!  "
        self.assertEqual(norm.normalize(raw), "Python Programming")

        raw_bool = " (python  OR  java)  AND  programming "
        self.assertEqual(norm.normalize(raw_bool), "(python OR java) AND programming")

        raw_phrase = ' "machine  learning" '
        self.assertEqual(norm.normalize(raw_phrase), '"machine learning"')

    def test_spell_checker_confidence(self):
        checker = self.pipeline.spell_checker
        
        # Exact match (confidence = 1.0, not modified)
        word, conf, was_fixed = checker.check_word("python")
        self.assertEqual(word, "python")
        self.assertEqual(conf, 1.0)
        self.assertFalse(was_fixed)

        # Typo match
        word_typo, conf_typo, was_fixed_typo = checker.check_word("pythn")
        self.assertEqual(word_typo, "python")
        self.assertGreaterEqual(conf_typo, config.SPELL_CORRECTION_THRESHOLD)
        self.assertTrue(was_fixed_typo)

    def test_synonym_expander(self):
        expander = self.pipeline.synonym_expander
        syns = expander.get_synonyms("programming", mode="conservative")
        self.assertIn("coding", syns)

        # Expanding list of terms
        expanded = expander.expand_query_terms(["programming"], mode="conservative")
        self.assertIn("coding", expanded)

    def test_intent_classifier(self):
        classifier = IntentClassifier()
        
        # Boolean intent
        intent1, strat1 = classifier.classify("python AND programming", ["python", "AND", "programming"])
        self.assertEqual(intent1, "boolean")
        self.assertEqual(strat1, "bm25")

        # Phrase intent
        intent2, strat2 = classifier.classify('"machine learning"', ['"machine learning"'])
        self.assertEqual(intent2, "phrase")
        self.assertEqual(strat2, "bm25")

        # Informational question intent
        intent3, strat3 = classifier.classify("how to learn programming concepts", ["how", "to", "learn", "programming", "concepts"])
        self.assertEqual(intent3, "informational")
        self.assertEqual(strat3, "semantic")

        # Short keyword intent
        intent4, strat4 = classifier.classify("python", ["python"])
        self.assertEqual(intent4, "keyword")
        self.assertEqual(strat4, "bm25")

    def test_query_suggester_autocomplete(self):
        suggester = self.pipeline.suggester
        suggestions = suggester.suggest("pyt", limit=3)
        self.assertIn("python", suggestions)

    def test_full_pipeline_analysis(self):
        repr_obj = self.pipeline.analyze("  pythn  AND  progrmming  ")
        self.assertEqual(repr_obj.intent, "boolean")
        self.assertEqual(repr_obj.suggested_strategy, "bm25")
        self.assertIn("python", repr_obj.effective_query)

        data = repr_obj.to_dict()
        self.assertIn("original_query", data)
        self.assertIn("normalized_query", data)
        self.assertIn("intent", data)

    def test_api_v1_suggest_endpoint(self):
        client = app.test_client()
        res = client.get("/api/v1/suggest?q=pyt&limit=3")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["prefix"], "pyt")
        self.assertIsInstance(data["suggestions"], list)


if __name__ == '__main__':
    unittest.main()
