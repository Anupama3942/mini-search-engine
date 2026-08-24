import unittest
import sys
from pathlib import Path

# Add the parent directory to sys.path to import search.py
sys.path.append(str(Path(__file__).parent.parent))

from search import (
    process_text,
    process_all_documents,
    build_inverted_index,
    calculate_score,
    search_index
)

class TestSearchRanking(unittest.TestCase):
    
    def setUp(self):
        # Sample documents for testing
        self.raw_documents = {
            "python.txt": "Python is a programming language. Python is easy to learn.",
            "web.txt": "Python can be used for web development. Web programming is useful.",
            "java.txt": "Java is a programming language used for application development.",
            "tie1.txt": "apple banana",
            "tie2.txt": "banana apple"
        }
        self.processed_docs = process_all_documents(self.raw_documents)
        self.index = build_inverted_index(self.processed_docs)

    def test_calculate_score(self):
        doc_tokens = ["python", "programming", "language", "python", "easy", "learn"]
        query_terms = ["python", "programming"]
        score = calculate_score(doc_tokens, query_terms)
        self.assertEqual(score, 3) # python=2, programming=1

    def test_ranking_order_single_term(self):
        # python.txt has 'python' twice, web.txt has it once, java.txt has it zero times
        results = search_index(self.index, self.processed_docs, "python")
        
        self.assertEqual(len(results), 2)
        # Check order
        self.assertEqual(results[0][0], "python.txt")
        self.assertEqual(results[0][1], 2)
        
        self.assertEqual(results[1][0], "web.txt")
        self.assertEqual(results[1][1], 1)

    def test_ranking_order_multi_term(self):
        # python programming
        # python.txt: python=2, programming=1 -> 3
        # web.txt: python=1, programming=1 -> 2
        # java.txt: python=0, programming=1 -> 1
        results = search_index(self.index, self.processed_docs, "python programming")
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "python.txt")
        self.assertEqual(results[0][1], 3)
        self.assertEqual(results[1][0], "web.txt")
        self.assertEqual(results[1][1], 2)
        self.assertEqual(results[2][0], "java.txt")
        self.assertEqual(results[2][1], 1)

    def test_tie_handling(self):
        # Both tie1.txt and tie2.txt have score 1 for 'apple'.
        # Tie should be broken alphabetically: tie1.txt before tie2.txt
        results = search_index(self.index, self.processed_docs, "apple")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "tie1.txt")
        self.assertEqual(results[0][1], 1)
        self.assertEqual(results[1][0], "tie2.txt")
        self.assertEqual(results[1][1], 1)

    def test_duplicate_query_terms(self):
        # "python python programming" should not inflate the score
        results_duplicate = search_index(self.index, self.processed_docs, "python python programming")
        results_normal = search_index(self.index, self.processed_docs, "python programming")
        self.assertEqual(results_duplicate, results_normal)

    def test_empty_query(self):
        results = search_index(self.index, self.processed_docs, "")
        self.assertEqual(results, [])

    def test_no_results(self):
        results = search_index(self.index, self.processed_docs, "blockchain")
        self.assertEqual(results, [])

    def test_stop_words(self):
        # 'the' is a stop word, 'python' should match
        results = search_index(self.index, self.processed_docs, "the python")
        # Should be same as just 'python'
        results_python = search_index(self.index, self.processed_docs, "python")
        self.assertEqual(results, results_python)

if __name__ == '__main__':
    unittest.main()
