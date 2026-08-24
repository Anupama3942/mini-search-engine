import unittest
import sys
import math
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from query_parser import tokenize_query, QueryParser

# A dummy process_text just for testing tokenization directly
def dummy_process_text(text):
    return [text.lower()]

class TestBooleanSearch(unittest.TestCase):
    
    def setUp(self):
        # Sample documents for testing
        self.raw_documents = {
            "doc1.txt": "python programming",
            "doc2.txt": "python java programming",
            "doc3.txt": "java programming",
            "doc4.txt": "java"
        }
        
        self.engine = SearchEngine()
        self.engine.documents = self.raw_documents
        self.engine.processed_documents = {}
        self.engine.inverted_index = {}
        self.engine._build_index()

    def test_tokenizer(self):
        tokens = tokenize_query("(Python OR java) AND programming", dummy_process_text)
        self.assertEqual(tokens, ['(', 'python', 'OR', 'java', ')', 'AND', 'programming'])
        
        # Test case insensitivity of operators
        tokens2 = tokenize_query("python and java or not programming", dummy_process_text)
        self.assertEqual(tokens2, ['python', 'AND', 'java', 'OR', 'NOT', 'programming'])

    def test_parser_invalid_syntax(self):
        # Using search engine which handles errors by returning dict
        res = self.engine.search("python AND")
        self.assertTrue(isinstance(res, dict))
        self.assertIn("error", res)
        
        res = self.engine.search("(")
        self.assertIn("error", res)

        res = self.engine.search('python "programming"')
        self.assertIn("error", res)

    def test_boolean_and(self):
        # python AND programming -> {doc1, doc2}
        results = self.engine.search("python AND programming")
        self.assertEqual(len(results), 2)
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc1.txt", "doc2.txt"})

    def test_boolean_or(self):
        # python OR java -> {doc1, doc2, doc3, doc4}
        results = self.engine.search("python OR java")
        self.assertEqual(len(results), 4)

    def test_boolean_not(self):
        # NOT java -> everything without java -> {doc1}
        results = self.engine.search("NOT java")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['filename'], "doc1.txt")

    def test_boolean_and_not(self):
        # python AND NOT java -> {doc1}
        results = self.engine.search("python AND NOT java")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['filename'], "doc1.txt")

    def test_parentheses(self):
        # (python OR java) AND programming
        # python OR java = {doc1, doc2, doc3, doc4}
        # AND programming = {doc1, doc2, doc3}
        results = self.engine.search("(python OR java) AND programming")
        self.assertEqual(len(results), 3)
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc1.txt", "doc2.txt", "doc3.txt"})

    def test_operator_precedence(self):
        # python OR java AND programming
        # AND has higher precedence, so: python OR (java AND programming)
        # java AND programming = {doc2, doc3}
        # python = {doc1, doc2}
        # Union = {doc1, doc2, doc3}
        results = self.engine.search("python OR java AND programming")
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc1.txt", "doc2.txt", "doc3.txt"})

    def test_unknown_term(self):
        # python AND blockchain -> empty
        results = self.engine.search("python AND blockchain")
        self.assertEqual(results, [])

        # python OR blockchain -> {doc1, doc2}
        results = self.engine.search("python OR blockchain")
        self.assertEqual(len(results), 2)

    def test_ranking_integration(self):
        # TF-IDF should still rank the filtered results
        # python AND programming
        results = self.engine.search("python AND programming")
        # doc1 has "python programming" (TF=1/2 each)
        # doc2 has "python java programming" (TF=1/3 each)
        # doc1 should score higher
        self.assertEqual(results[0]['filename'], "doc1.txt")
        self.assertEqual(results[1]['filename'], "doc2.txt")
        # Ensure scores are > 0
        self.assertTrue(results[0]['score'] > 0)

if __name__ == '__main__':
    unittest.main()
