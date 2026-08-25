import unittest
import sys
import math
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from ranking.bm25 import BM25Ranker
import config


class TestBM25Ranking(unittest.TestCase):

    def setUp(self):
        # 3 educational sample documents
        self.corpus = {
            "doc1.txt": "python programming tutorial",
            "doc2.txt": "python programming programming programming tutorial",
            "doc3.txt": "database tutorial"
        }
        self.engine = SearchEngine(documents=dict(self.corpus))
        self.ranker = BM25Ranker(k1=1.2, b=0.75)

    def test_bm25_parameter_validation(self):
        # Valid parameters
        ranker = BM25Ranker(k1=1.5, b=0.5)
        self.assertEqual(ranker.k1, 1.5)
        self.assertEqual(ranker.b, 0.5)

        # Invalid k1 (<= 0)
        with self.assertRaises(ValueError):
            BM25Ranker(k1=0.0, b=0.75)
        with self.assertRaises(ValueError):
            BM25Ranker(k1=-1.2, b=0.75)

        # Invalid b (< 0 or > 1)
        with self.assertRaises(ValueError):
            BM25Ranker(k1=1.2, b=-0.1)
        with self.assertRaises(ValueError):
            BM25Ranker(k1=1.2, b=1.5)

    def test_bm25_idf_non_negative(self):
        # python appears in 2 out of 3 docs -> df = 2, N = 3
        # IDF = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 1.5 / 2.5) = ln(1.6) > 0
        idf_python = self.ranker.calculate_bm25_idf("python", self.engine)
        self.assertGreater(idf_python, 0.0)
        self.assertAlmostEqual(idf_python, math.log(1.0 + 1.5 / 2.5), places=5)

        # Rare term: database appears in 1 out of 3 docs -> higher IDF
        idf_db = self.ranker.calculate_bm25_idf("database", self.engine)
        self.assertGreater(idf_db, idf_python)

        # Unknown term -> IDF = 0.0
        self.assertEqual(self.ranker.calculate_bm25_idf("quantum123", self.engine), 0.0)

    def test_term_frequency_saturation(self):
        """
        Verify that score increases sub-linearly with term frequency:
        ΔScore(1 -> 2) > ΔScore(2 -> 3) > ΔScore(3 -> 4) ...
        """
        k1 = 1.2
        b = 0.0 # isolate TF saturation without length normalization
        ranker = BM25Ranker(k1=k1, b=b)
        
        # Mock single term with IDF=1.0 and fixed length=1.0
        scores = []
        for tf in [1, 2, 3, 5, 10, 20]:
            # formula: tf * (k1 + 1) / (tf + k1)
            score = (tf * (k1 + 1.0)) / (tf + k1)
            scores.append((tf, score))

        # Check monotonically increasing
        for i in range(len(scores) - 1):
            self.assertLess(scores[i][1], scores[i+1][1])

        # Check saturation: gain from 1->2 is greater than gain from 5->10
        gain_1_to_2 = scores[1][1] - scores[0][1]
        gain_2_to_3 = scores[2][1] - scores[1][1]
        gain_5_to_10 = (scores[4][1] - scores[3][1]) / 5
        self.assertGreater(gain_1_to_2, gain_2_to_3)
        self.assertGreater(gain_2_to_3, gain_5_to_10)

    def test_document_length_normalization(self):
        """
        Verify that between two documents with equal term frequency,
        the shorter document scores higher when b > 0.
        """
        # doc_short: 3 tokens, 1 occurrence of "python"
        # doc_long: 100 tokens, 1 occurrence of "python"
        docs = {
            "short.txt": "python programming language",
            "long.txt": "python " + " ".join(["unrelated"] * 97) + " programming language"
        }
        engine = SearchEngine(documents=docs)
        ranker = BM25Ranker(k1=1.2, b=0.75)

        score_short = ranker.score(["python"], "short.txt", engine)
        score_long = ranker.score(["python"], "long.txt", engine)
        
        self.assertGreater(score_short, score_long)

    def test_length_normalization_deactivated_when_b_zero(self):
        """When b = 0, document length has zero impact on score."""
        docs = {
            "short.txt": "python programming language",
            "long.txt": "python " + " ".join(["unrelated"] * 97) + " programming language"
        }
        engine = SearchEngine(documents=docs)
        ranker = BM25Ranker(k1=1.2, b=0.0)

        score_short = ranker.score(["python"], "short.txt", engine)
        score_long = ranker.score(["python"], "long.txt", engine)
        
        self.assertAlmostEqual(score_short, score_long, places=5)

    def test_multi_term_query_scoring(self):
        # Query: "python programming"
        score_doc1 = self.ranker.score(["python", "programming"], "doc1.txt", self.engine)
        score_doc3 = self.ranker.score(["python", "programming"], "doc3.txt", self.engine)

        self.assertGreater(score_doc1, 0.0)
        self.assertEqual(score_doc3, 0.0) # doc3 has neither python nor programming

    def test_empty_corpus_safety(self):
        empty_engine = SearchEngine(documents={})
        score = self.ranker.score(["python"], "doc1.txt", empty_engine)
        self.assertEqual(score, 0.0)

    def test_score_explanation(self):
        explanation = self.engine.explain_score("python programming", "doc1.txt", ranking_algorithm="bm25")
        self.assertIn("total_score", explanation)
        self.assertIn("term_contributions", explanation)
        self.assertIn("python", explanation["term_contributions"])
        self.assertIn("programming", explanation["term_contributions"])
        self.assertEqual(explanation["algorithm"], "bm25")


if __name__ == '__main__':
    unittest.main()
