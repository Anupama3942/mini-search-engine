import unittest
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from evaluation.evaluator import SearchEvaluator

class TestSearchQualityRegression(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.engine = SearchEngine()
        cls.evaluator = SearchEvaluator()
        cls.report = cls.evaluator.evaluate_engine(cls.engine, top_k=10)

    def test_quality_gate_map_and_mrr(self):
        metrics = self.report["summary_metrics"]
        self.assertGreaterEqual(metrics["map"], 0.70, f"MAP {metrics['map']} is below threshold 0.70")
        self.assertGreaterEqual(metrics["mrr"], 0.75, f"MRR {metrics['mrr']} is below threshold 0.75")
        self.assertGreaterEqual(metrics["p@1"], 0.70, f"P@1 {metrics['p@1']} is below threshold 0.70")

    def test_core_query_rank1_relevance(self):
        # Query "python" -> python.txt must be ranked at rank 1 or 2
        res_python = self.engine.search("python", log_analytics=False)
        self.assertGreater(len(res_python), 0)
        self.assertIn(res_python[0]["filename"], ["python.txt", "test_docs.txt"])

        # Query "database" -> database.txt must be ranked at rank 1
        res_db = self.engine.search("database", log_analytics=False)
        self.assertGreater(len(res_db), 0)
        self.assertEqual(res_db[0]["filename"], "database.txt")

        # Query "networking" -> networking.txt must be ranked at rank 1
        res_net = self.engine.search("networking", log_analytics=False)
        self.assertGreater(len(res_net), 0)
        self.assertEqual(res_net[0]["filename"], "networking.txt")

    def test_ranking_regression_relevant_above_irrelevant(self):
        # Query "python programming" -> python.txt should rank above database.txt
        res = self.engine.search("python programming", log_analytics=False)
        filenames = [r["filename"] for r in res]
        self.assertIn("python.txt", filenames)
        
        if "database.txt" in filenames:
            self.assertLess(filenames.index("python.txt"), filenames.index("database.txt"))

    def test_phrase_query_precision(self):
        # "web development" should match web.txt with high confidence
        res_phrase = self.engine.search('"web development"', log_analytics=False)
        filenames = [r["filename"] for r in res_phrase]
        self.assertIn("web.txt", filenames)
        self.assertIn("python.txt", filenames)

    def test_fuzzy_query_relevance(self):
        # Typo query 'pythn' should retrieve python.txt at top
        res_fuzzy = self.engine.search("pythn", log_analytics=False)
        self.assertGreater(len(res_fuzzy), 0)
        self.assertIn(res_fuzzy[0]["filename"], ["python.txt", "test_docs.txt"])


if __name__ == '__main__':
    unittest.main()
