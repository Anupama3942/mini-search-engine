import unittest
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from ranking import get_ranker, BM25Ranker, TFIDFRanker, FrequencyRanker

class TestRankingStrategies(unittest.TestCase):

    def setUp(self):
        self.corpus = {
            "doc1.txt": "python programming language with python examples",
            "doc2.txt": "python is simple and clean",
            "doc3.txt": "java programming language with enterprise backend systems"
        }
        self.engine = SearchEngine(documents=dict(self.corpus))

    def test_ranker_factory(self):
        r_bm25 = get_ranker("bm25", k1=1.5, b=0.6)
        self.assertIsInstance(r_bm25, BM25Ranker)
        self.assertEqual(r_bm25.k1, 1.5)
        self.assertEqual(r_bm25.b, 0.6)

        r_tfidf = get_ranker("tfidf")
        self.assertIsInstance(r_tfidf, TFIDFRanker)

        r_freq = get_ranker("frequency")
        self.assertIsInstance(r_freq, FrequencyRanker)

        with self.assertRaises(ValueError):
            get_ranker("non_existent_algorithm")

    def test_search_engine_ranking_algorithm_switching(self):
        # 1. Search with BM25
        res_bm25 = self.engine.search("python", ranking_algorithm="bm25", log_analytics=False)
        self.assertEqual(res_bm25.ranking_algorithm, "bm25")
        self.assertGreater(len(res_bm25), 0)

        # 2. Search with TF-IDF
        res_tfidf = self.engine.search("python", ranking_algorithm="tfidf", log_analytics=False)
        self.assertEqual(res_tfidf.ranking_algorithm, "tfidf")
        self.assertGreater(len(res_tfidf), 0)

        # 3. Search with Frequency
        res_freq = self.engine.search("python", ranking_algorithm="frequency", log_analytics=False)
        self.assertEqual(res_freq.ranking_algorithm, "frequency")
        self.assertGreater(len(res_freq), 0)

    def test_incremental_indexing_updates_bm25_stats(self):
        initial_avg_len = self.engine.index_stats["avg_document_length"]
        initial_doc_count = self.engine.index_stats["total_documents"]

        # Add document
        self.engine.add_document("doc4.txt", "rust systems programming language is fast and modern")
        self.assertEqual(self.engine.index_stats["total_documents"], initial_doc_count + 1)
        self.assertNotEqual(self.engine.index_stats["avg_document_length"], 0.0)

        # Remove document
        self.engine.remove_document("doc4.txt")
        self.assertEqual(self.engine.index_stats["total_documents"], initial_doc_count)
        self.assertEqual(self.engine.index_stats["avg_document_length"], initial_avg_len)


if __name__ == '__main__':
    unittest.main()
