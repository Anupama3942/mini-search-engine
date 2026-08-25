import unittest
import sys
import tempfile
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from evaluation.tuner import BM25Tuner

class TestBM25Tuning(unittest.TestCase):

    def test_bm25_tuner_execution(self):
        engine = SearchEngine()
        tuner = BM25Tuner()

        # Run small fast grid
        results = tuner.tune(engine, k1_values=[1.0, 1.5], b_values=[0.5, 0.75])
        self.assertEqual(results["total_configurations_tested"], 4)
        self.assertIn("best_configuration", results)
        self.assertIn("k1", results["best_configuration"])
        self.assertIn("b", results["best_configuration"])
        self.assertGreater(results["best_configuration"]["map"], 0.70)

        # Test report export
        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "test_tuning.json"
            self.assertTrue(tuner.export_tuning_report(results, out_file))
            self.assertTrue(out_file.exists())


if __name__ == '__main__':
    unittest.main()
