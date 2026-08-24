import unittest
import sys
import math
from pathlib import Path

# Add the parent directory to sys.path to import search.py
sys.path.append(str(Path(__file__).parent.parent))

from search import (
    process_all_documents,
    build_inverted_index,
    calculate_tf,
    calculate_document_frequency,
    calculate_idf,
    calculate_tfidf,
    score_document,
    search_index
)

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
        self.processed_docs = process_all_documents(self.raw_documents)
        self.index = build_inverted_index(self.processed_docs)
        self.total_docs = len(self.processed_docs) # 6

    def test_calculate_tf(self):
        doc_tokens = ["python", "programming", "python"]
        # python appears 2 out of 3 times -> 2/3
        self.assertAlmostEqual(calculate_tf("python", doc_tokens), 2/3)
        # programming appears 1 out of 3 times -> 1/3
        self.assertAlmostEqual(calculate_tf("programming", doc_tokens), 1/3)
        
        # Empty document test
        self.assertEqual(calculate_tf("python", []), 0.0)

    def test_calculate_df(self):
        # 'python' is in python.txt and web.txt (2 documents)
        self.assertEqual(calculate_document_frequency("python", self.index), 2)
        # 'java' is in java.txt (1 document)
        self.assertEqual(calculate_document_frequency("java", self.index), 1)
        # 'unknown' is in 0 documents
        self.assertEqual(calculate_document_frequency("unknown", self.index), 0)

    def test_calculate_idf(self):
        # python df = 2, total = 6 -> log(6/2) = log(3)
        self.assertAlmostEqual(calculate_idf("python", self.index, self.total_docs), math.log(3))
        # java df = 1, total = 6 -> log(6/1) = log(6)
        self.assertAlmostEqual(calculate_idf("java", self.index, self.total_docs), math.log(6))
        # unknown df = 0, should return 0.0 to prevent crash
        self.assertEqual(calculate_idf("unknown", self.index, self.total_docs), 0.0)

    def test_common_vs_rare_term(self):
        # Verify common term has lower IDF than rare term
        # 'programming' appears in 2 docs, 'java' in 1
        idf_prog = calculate_idf("programming", self.index, self.total_docs)
        idf_java = calculate_idf("java", self.index, self.total_docs)
        self.assertTrue(idf_prog < idf_java)

    def test_calculate_tfidf(self):
        # python in python.txt (doc_tokens = ["python", "programming", "python"])
        # TF = 2/3
        # IDF = log(6/2) = log(3)
        expected_tfidf = (2/3) * math.log(3)
        actual_tfidf = calculate_tfidf("python", self.processed_docs["python.txt"], self.index, self.total_docs)
        self.assertAlmostEqual(expected_tfidf, actual_tfidf)

    def test_score_document(self):
        # Query: "python programming" on python.txt
        # TF-IDF(python) = (2/3) * log(3)
        # TF-IDF(programming) = (1/3) * log(6/2) = (1/3) * log(3)
        expected_score = ((2/3) * math.log(3)) + ((1/3) * math.log(3))
        actual_score = score_document(
            self.processed_docs["python.txt"], 
            ["python", "programming"], 
            self.index, 
            self.total_docs
        )
        self.assertAlmostEqual(expected_score, actual_score)

    def test_ranking_order_single_term(self):
        # query 'python'
        # python.txt -> TF = 2/3
        # web.txt -> TF = 1/2
        # IDF is the same for both. So python.txt should rank higher.
        results = search_index(self.index, self.processed_docs, "python")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "python.txt")
        self.assertEqual(results[1][0], "web.txt")

    def test_tie_handling(self):
        # tie1.txt and tie2.txt have score for 'apple'.
        # Tie should be broken alphabetically: tie1.txt before tie2.txt
        results = search_index(self.index, self.processed_docs, "apple")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "tie1.txt")
        self.assertEqual(results[1][0], "tie2.txt")

    def test_duplicate_query_terms(self):
        results_duplicate = search_index(self.index, self.processed_docs, "python python")
        results_normal = search_index(self.index, self.processed_docs, "python")
        self.assertEqual(results_duplicate, results_normal)

    def test_no_results(self):
        results = search_index(self.index, self.processed_docs, "blockchain")
        self.assertEqual(results, [])

    def test_punctuation_and_case(self):
        results_upper = search_index(self.index, self.processed_docs, "PYTHON!")
        results_lower = search_index(self.index, self.processed_docs, "python")
        self.assertEqual(results_upper, results_lower)


if __name__ == '__main__':
    unittest.main()
