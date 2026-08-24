import unittest
import sys
import math
from pathlib import Path

# Add the parent directory to sys.path to import search.py
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine, process_text

class TestSearchRankingTFIDF(unittest.TestCase):
    
    def setUp(self):
        # Sample documents for testing
        self.raw_documents = {
            "python.txt": "python programming python",
            "java.txt": "java programming",
            "web.txt": "python web",
            "tie1.txt": "apple banana",
            "tie2.txt": "banana apple",
            "empty.txt": ""
        }
        
        self.engine = SearchEngine()
        # Mock the documents
        self.engine.documents = self.raw_documents
        self.engine.processed_documents = {}
        self.engine.inverted_index = {}
        self.engine._build_index()

    def test_calculate_tf(self):
        doc_tokens = ["python", "programming", "python"]
        # python appears 2 out of 3 times -> 2/3
        self.assertAlmostEqual(self.engine.calculate_tf("python", doc_tokens), 2/3)
        # programming appears 1 out of 3 times -> 1/3
        self.assertAlmostEqual(self.engine.calculate_tf("programming", doc_tokens), 1/3)
        
        # Empty document test
        self.assertEqual(self.engine.calculate_tf("python", []), 0.0)

    def test_calculate_idf(self):
        # python df = 2, total = 6 -> log(6/2) = log(3)
        self.assertAlmostEqual(self.engine.calculate_idf("python"), math.log(3))
        # java df = 1, total = 6 -> log(6/1) = log(6)
        self.assertAlmostEqual(self.engine.calculate_idf("java"), math.log(6))
        # unknown df = 0, should return 0.0 to prevent crash
        self.assertEqual(self.engine.calculate_idf("unknown"), 0.0)

    def test_common_vs_rare_term(self):
        # Verify common term has lower IDF than rare term
        # 'programming' appears in 2 docs, 'java' in 1
        idf_prog = self.engine.calculate_idf("programming")
        idf_java = self.engine.calculate_idf("java")
        self.assertTrue(idf_prog < idf_java)

    def test_score_document(self):
        # Query: "python programming" on python.txt
        # TF-IDF(python) = (2/3) * log(3)
        # TF-IDF(programming) = (1/3) * log(6/2) = (1/3) * log(3)
        expected_score = ((2/3) * math.log(3)) + ((1/3) * math.log(3))
        
        results = self.engine.search("python programming")
        
        # Find score for python.txt
        score = next(r['score'] for r in results if r['filename'] == "python.txt")
        self.assertAlmostEqual(expected_score, score)

    def test_ranking_order_single_term(self):
        # query 'python'
        # python.txt -> TF = 2/3
        # web.txt -> TF = 1/2
        # IDF is the same for both. So python.txt should rank higher.
        results = self.engine.search("python")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["filename"], "python.txt")
        self.assertEqual(results[1]["filename"], "web.txt")

    def test_tie_handling(self):
        # tie1.txt and tie2.txt have score for 'apple'.
        # Tie should be broken alphabetically: tie1.txt before tie2.txt
        results = self.engine.search("apple")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["filename"], "tie1.txt")
        self.assertEqual(results[1]["filename"], "tie2.txt")

    def test_duplicate_query_terms(self):
        results_duplicate = self.engine.search("python python")
        results_normal = self.engine.search("python")
        
        # Just check filenames and scores are equal
        self.assertEqual([(r["filename"], r["score"]) for r in results_duplicate],
                         [(r["filename"], r["score"]) for r in results_normal])

    def test_no_results(self):
        results = self.engine.search("blockchain")
        self.assertEqual(results, [])

    def test_punctuation_and_case(self):
        results_upper = self.engine.search("PYTHON!")
        results_lower = self.engine.search("python")
        self.assertEqual([(r["filename"], r["score"]) for r in results_upper],
                         [(r["filename"], r["score"]) for r in results_lower])

if __name__ == '__main__':
    unittest.main()
