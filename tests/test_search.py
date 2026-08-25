import unittest
import sys
import math
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from query_parser import tokenize_query

# A dummy process_text just for testing tokenization directly
def dummy_process_text(text):
    return text.lower().split()

class TestSearchEngine(unittest.TestCase):
    
    def setUp(self):
        # Sample documents for testing
        self.raw_documents = {
            "doc1.txt": "python programming language",
            "doc2.txt": "python is a programming language",
            "doc3.txt": "programming python language",
            "doc4.txt": "machine learning algorithm",
            "doc5.txt": "machine algorithm learning",
            "doc6.txt": "python programming is powerful. python programming is fun.",
            "doc7.txt": "java programming javascript"
        }
        
        self.engine = SearchEngine()
        self.engine.documents = self.raw_documents
        self.engine.processed_documents = {}
        self.engine.inverted_index = {}
        self.engine.positional_index = {}
        self.engine.fuzzy_cache = {}
        self.engine._build_index()

    # --- Positional Index & Phrase Tests ---
    def test_positional_index_creation(self):
        self.assertEqual(self.engine.positional_index["python"]["doc1.txt"], [0])
        self.assertEqual(self.engine.positional_index["programming"]["doc1.txt"], [1])

    def test_exact_phrase_match(self):
        results = self.engine.search('"python programming"')
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc1.txt", "doc2.txt", "doc6.txt"})

    def test_wrong_order(self):
        results = self.engine.search('"programming python"')
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc3.txt"})

    def test_phrase_compatibility_remains_exact(self):
        # "machne learning" has a typo inside quotes -> should NOT match doc4 because phrases remain exact!
        results = self.engine.search('"machne learning"')
        self.assertEqual(len(results), 0)

    # --- Boolean Tests ---
    def test_boolean_and(self):
        results = self.engine.search("python AND java")
        self.assertEqual(len(results), 0)

        results = self.engine.search("python AND programming")
        self.assertEqual(len(results), 4)

    def test_boolean_or(self):
        results = self.engine.search("python OR java")
        filenames = {r['filename'] for r in results}
        self.assertIn("doc7.txt", filenames)
        self.assertIn("doc1.txt", filenames)

    def test_boolean_and_not(self):
        results = self.engine.search("programming AND NOT java")
        filenames = {r['filename'] for r in results}
        self.assertNotIn("doc7.txt", filenames)

    # --- Fuzzy Search & Typo Tolerance Tests ---
    def test_fuzzy_search_single_typo(self):
        # 'pythn' should resolve to 'python'
        results = self.engine.search("pythn")
        self.assertGreater(len(results), 0)
        self.assertEqual(results.did_you_mean, "python")
        self.assertEqual(results.original_query, "pythn")
        filenames = {r['filename'] for r in results}
        self.assertIn("doc1.txt", filenames)

    def test_fuzzy_search_exact_match_no_correction(self):
        # 'python' is exact -> did_you_mean should be None
        results = self.engine.search("python")
        self.assertGreater(len(results), 0)
        self.assertIsNone(results.did_you_mean)

    def test_fuzzy_search_multiple_typos_boolean(self):
        # 'pythn AND programing' -> 'python AND programming'
        results = self.engine.search("pythn AND programing")
        self.assertGreater(len(results), 0)
        self.assertEqual(results.did_you_mean, "python AND programming")
        filenames = {r['filename'] for r in results}
        self.assertIn("doc1.txt", filenames)
        self.assertIn("doc2.txt", filenames)

    def test_fuzzy_search_or(self):
        # 'pythn OR java' -> 'python OR java'
        results = self.engine.search("pythn OR java")
        self.assertEqual(results.did_you_mean, "python OR java")
        filenames = {r['filename'] for r in results}
        self.assertIn("doc1.txt", filenames)
        self.assertIn("doc7.txt", filenames)

    def test_fuzzy_search_not(self):
        # 'python AND NOT javscript' -> 'python AND NOT javascript'
        # doc7 has javascript, so excluded
        results = self.engine.search("python AND NOT javscript")
        self.assertEqual(results.did_you_mean, "python AND NOT javascript")
        filenames = {r['filename'] for r in results}
        self.assertNotIn("doc7.txt", filenames)

    def test_fuzzy_search_parentheses(self):
        # '(pythn OR java) AND programing' -> '(python OR java) AND programming'
        results = self.engine.search("(pythn OR java) AND programing")
        self.assertEqual(results.did_you_mean, "(python OR java) AND programming")
        self.assertGreater(len(results), 0)

    def test_fuzzy_search_unknown_term(self):
        results = self.engine.search("qwertyuiopasdf")
        self.assertEqual(len(results), 0)
        self.assertIsNone(results.did_you_mean)

    def test_fuzzy_cache_cleared_on_rebuild(self):
        self.engine.search("pythn")
        self.assertIn("pythn", self.engine.fuzzy_cache)
        self.engine._build_index()
        self.assertEqual(len(self.engine.fuzzy_cache), 0)


if __name__ == '__main__':
    unittest.main()
