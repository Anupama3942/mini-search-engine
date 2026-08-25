import unittest
import sys
import tempfile
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from evaluation.metrics import (
    precision,
    recall,
    f1_score,
    precision_at_k,
    recall_at_k,
    average_precision,
    reciprocal_rank,
    mean_average_precision,
    mean_reciprocal_rank,
    calculate_confusion_matrix
)
from evaluation.evaluator import SearchEvaluator, validate_evaluation_dataset


class TestEvaluationMetrics(unittest.TestCase):
    
    def test_precision_recall_f1(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc5"]
        
        # 3 TP out of 5 retrieved -> Precision = 0.60
        self.assertAlmostEqual(precision(retrieved, relevant), 0.60)
        
        # 3 TP out of 3 relevant -> Recall = 1.00
        self.assertAlmostEqual(recall(retrieved, relevant), 1.00)
        
        # F1 = 2 * (0.6 * 1.0) / (0.6 + 1.0) = 1.2 / 1.6 = 0.75
        self.assertAlmostEqual(f1_score(0.6, 1.0), 0.75)

    def test_precision_recall_edge_cases(self):
        # 0 retrieved, 0 relevant
        self.assertEqual(precision([], []), 1.0)
        self.assertEqual(recall([], []), 1.0)
        
        # 0 retrieved, 3 relevant
        self.assertEqual(precision([], ["doc1", "doc2"]), 0.0)
        self.assertEqual(recall([], ["doc1", "doc2"]), 0.0)

        # F1 division by zero
        self.assertEqual(f1_score(0.0, 0.0), 0.0)

    def test_precision_and_recall_at_k(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3"] # 2 relevant total

        self.assertEqual(precision_at_k(retrieved, relevant, 1), 1.0)  # 1/1
        self.assertEqual(precision_at_k(retrieved, relevant, 2), 0.5)  # 1/2
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 3), 2/3, places=3)  # 2/3
        self.assertEqual(precision_at_k(retrieved, relevant, 5), 0.4)  # 2/5

        self.assertEqual(recall_at_k(retrieved, relevant, 1), 0.5)    # 1/2
        self.assertEqual(recall_at_k(retrieved, relevant, 3), 1.0)    # 2/2

    def test_average_precision_hand_calculation(self):
        # Rank 1: rel (P@1 = 1/1 = 1.0)
        # Rank 2: non (P@2 = 1/2 = 0.5)
        # Rank 3: rel (P@3 = 2/3 = 0.6667)
        # Rank 4: non (P@4 = 2/4 = 0.5)
        # Rank 5: rel (P@5 = 3/5 = 0.6)
        # Total relevant = 3
        # AP = (1.0 + 0.6667 + 0.6) / 3 = 2.2667 / 3 ≈ 0.7556
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = ["d1", "d3", "d5"]
        
        expected_ap = round((1.0 + (2/3) + (3/5)) / 3, 4)
        actual_ap = average_precision(retrieved, relevant)
        self.assertAlmostEqual(actual_ap, expected_ap, places=3)

    def test_reciprocal_rank(self):
        # 1st relevant at rank 1 -> RR = 1.0
        self.assertEqual(reciprocal_rank(["d1", "d2"], ["d1"]), 1.0)
        
        # 1st relevant at rank 2 -> RR = 0.5
        self.assertEqual(reciprocal_rank(["d2", "d1"], ["d1"]), 0.5)
        
        # 1st relevant at rank 4 -> RR = 0.25
        self.assertEqual(reciprocal_rank(["d2", "d3", "d4", "d1"], ["d1"]), 0.25)
        
        # No relevant found -> RR = 0.0
        self.assertEqual(reciprocal_rank(["d2", "d3"], ["d1"]), 0.0)

    def test_mean_macro_metrics(self):
        ap_list = [1.0, 0.5, 0.75]
        self.assertEqual(mean_average_precision(ap_list), 0.75)
        
        rr_list = [1.0, 0.5, 0.0]
        self.assertAlmostEqual(mean_reciprocal_rank(rr_list), 0.5)

    def test_confusion_matrix(self):
        all_docs = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2", "doc3", "doc4"]
        
        cm = calculate_confusion_matrix(retrieved, relevant, all_docs)
        self.assertEqual(cm["true_positives"], 2)   # doc2, doc3
        self.assertEqual(cm["false_positives"], 1)  # doc1
        self.assertEqual(cm["false_negatives"], 1)  # doc4
        self.assertEqual(cm["true_negatives"], 2)   # doc5, doc6

    def test_dataset_validation(self):
        val = validate_evaluation_dataset()
        self.assertTrue(val["is_valid"])
        self.assertEqual(val["error_count"], 0)
        self.assertGreater(val["query_count"], 0)

    def test_search_evaluator(self):
        engine = SearchEngine()
        evaluator = SearchEvaluator()
        
        report = evaluator.evaluate_engine(engine, top_k=5)
        self.assertIn("summary_metrics", report)
        self.assertIn("map", report["summary_metrics"])
        self.assertIn("mrr", report["summary_metrics"])
        self.assertGreaterEqual(report["summary_metrics"]["map"], 0.70)
        self.assertGreaterEqual(report["summary_metrics"]["mrr"], 0.75)

    def test_export_report_files(self):
        engine = SearchEngine()
        evaluator = SearchEvaluator()
        report = evaluator.evaluate_engine(engine, top_k=5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "test_report.json"
            csv_path = Path(temp_dir) / "test_report.csv"
            
            self.assertTrue(evaluator.export_report_json(report, json_path))
            self.assertTrue(evaluator.export_report_csv(report, csv_path))
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())


if __name__ == '__main__':
    unittest.main()
