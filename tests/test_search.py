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

class TestPhraseSearch(unittest.TestCase):
    
    def setUp(self):
        # Sample documents for testing
        self.raw_documents = {
            "doc1.txt": "python programming language",
            "doc2.txt": "python is a programming language", # 'is a' might be removed by stop words
            "doc3.txt": "programming python language",
            "doc4.txt": "machine learning algorithm",
            "doc5.txt": "machine algorithm learning",
            "doc6.txt": "python programming is powerful. python programming is fun.",
            "doc7.txt": "java programming"
        }
        
        self.engine = SearchEngine()
        self.engine.documents = self.raw_documents
        self.engine.processed_documents = {}
        self.engine.inverted_index = {}
        self.engine.positional_index = {}
        self.engine._build_index()

    def test_positional_index_creation(self):
        # doc1: "python programming language"
        # Since no stop words, positions are 0, 1, 2
        self.assertEqual(self.engine.positional_index["python"]["doc1.txt"], [0])
        self.assertEqual(self.engine.positional_index["programming"]["doc1.txt"], [1])
        
        # doc6: "python programming is powerful. python programming is fun."
        # Stop words 'is' removed.
        # Original: python(0), programming(1), is(stop), powerful(2), python(3), programming(4), is(stop), fun(5)
        # So python -> [0, 3], programming -> [1, 4]
        self.assertEqual(self.engine.positional_index["python"]["doc6.txt"], [0, 3])
        self.assertEqual(self.engine.positional_index["programming"]["doc6.txt"], [1, 4])

    def test_tokenizer_phrases(self):
        tokens = tokenize_query('python AND "machine learning"', dummy_process_text)
        self.assertEqual(tokens, ['python', 'AND', ('PHRASE', ['machine', 'learning'])])
        
        # Test unclosed quotes
        with self.assertRaises(ValueError):
            tokenize_query('"machine learning', dummy_process_text)
            
        # Test empty phrase
        with self.assertRaises(ValueError):
            tokenize_query('""', dummy_process_text)

    def test_exact_phrase_match(self):
        # doc1: "python programming language" -> Matches
        # doc2: "python is a programming language" -> Stop words removed -> "python programming language" -> Matches!
        # doc3: "programming python language" -> Reversed -> NO MATCH
        results = self.engine.search('"python programming"')
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc1.txt", "doc2.txt", "doc6.txt"})

    def test_wrong_order(self):
        # "programming python" should match doc3 only
        results = self.engine.search('"programming python"')
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc3.txt"})

    def test_multiple_words_phrase(self):
        # "machine learning algorithm"
        results = self.engine.search('"machine learning algorithm"')
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc4.txt"})
        
        # doc5 has "machine algorithm learning" so it should not match
        self.assertNotIn("doc5.txt", filenames)

    def test_boolean_and_phrase(self):
        # "machine learning" AND algorithm -> doc4
        results = self.engine.search('"machine learning" AND algorithm')
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc4.txt"})

    def test_phrase_and_not(self):
        # "python programming" AND NOT powerful
        # doc1 and doc2 match phrase but not powerful. doc6 has powerful, so excluded.
        results = self.engine.search('"python programming" AND NOT powerful')
        filenames = {r['filename'] for r in results}
        self.assertEqual(filenames, {"doc1.txt", "doc2.txt"})

    def test_unknown_phrase(self):
        # "quantum blockchain unicorn"
        results = self.engine.search('"quantum blockchain unicorn"')
        self.assertEqual(results, [])

    def test_tf_idf_positive_terms(self):
        # Ranking for "python programming" should use TF-IDF of 'python' and 'programming'
        # doc6 has them twice, doc1 has them once. doc6 should rank higher.
        results = self.engine.search('"python programming"')
        
        # Assuming length normalization works properly, let's just ensure doc6 gets a valid score
        # doc1 length = 3. doc6 length = 6. 
        # TF for doc1 = 1/3 (python), 1/3 (prog). 
        # TF for doc6 = 2/6 (python), 2/6 (prog). 
        # Actually, TF is exactly equal! 1/3 == 2/6. So scores will be tied!
        # When tied, it falls back to alphabetical filename sort.
        # doc1.txt comes before doc6.txt
        filenames = [r['filename'] for r in results]
        self.assertIn("doc1.txt", filenames)
        self.assertIn("doc6.txt", filenames)
        
        # Verify scores are > 0
        self.assertTrue(results[0]['score'] > 0)


if __name__ == '__main__':
    unittest.main()
