import unittest
import sys
import json
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from experimentation import (
    Experiment,
    ExperimentRegistry,
    compare_variants,
    calculate_mean,
    calculate_variance,
    OfflineABEvaluator
)
from analytics import record_click, get_ctr_analytics, record_search
from app import app
import config


class TestExperimentationAndAnalytics(unittest.TestCase):

    def setUp(self):
        self.registry = ExperimentRegistry.get_instance()
        self.client = app.test_client()

    def test_deterministic_variant_assignment(self):
        exp = Experiment(
            id="test_exp",
            name="Test Experiment",
            description="Testing deterministic assignment",
            enabled=True,
            traffic_percentage=100.0,
            variant_a_method="bm25",
            variant_b_method="hybrid",
            split_ratio=0.50
        )

        # Same entity must always receive the same variant
        assignment1 = exp.assign_variant("user_session_123")
        assignment2 = exp.assign_variant("user_session_123")
        self.assertEqual(assignment1, assignment2)
        self.assertIn(assignment1[0], ("A", "B"))

    def test_experiment_registry_fallback(self):
        # Invalid experiment falls back to DEFAULT_RANKING_ALGORITHM
        var, method = self.registry.assign("non_existent_exp", "session_abc")
        self.assertIsNone(var)
        self.assertEqual(method, config.DEFAULT_RANKING_ALGORITHM)

    def test_statistical_comparison_calculation(self):
        # Known arrays
        a = [1.0, 0.9, 0.95, 0.88, 0.92, 0.96, 0.94, 0.91, 0.89, 0.93]
        b = [0.8, 0.75, 0.82, 0.78, 0.79, 0.81, 0.77, 0.83, 0.76, 0.80]

        res = compare_variants(a, b, metric_name="NDCG@5")
        self.assertEqual(res["sample_size_a"], 10)
        self.assertEqual(res["sample_size_b"], 10)
        self.assertLess(res["difference"], 0)
        self.assertTrue(res["statistically_significant"])
        self.assertEqual(res["conclusion"], "Variant A is significantly better")

    def test_offline_ab_evaluator(self):
        evaluator = OfflineABEvaluator()
        exp = self.registry.get("bm25_vs_hybrid")
        self.assertIsNotNone(exp)
        
        report = evaluator.run_offline_experiment(exp)
        self.assertEqual(report["experiment_id"], "bm25_vs_hybrid")
        self.assertIn("statistical_analysis", report)
        self.assertIn("NDCG@5", report["statistical_analysis"])

    def test_click_recording_and_ctr_analytics(self):
        record_search(
            query="python",
            result_count=4,
            search_duration=0.005,
            search_method="bm25",
            request_id="test_req_1"
        )
        record_click(
            request_id="test_req_1",
            doc_id="python.txt",
            position=1,
            search_method="bm25"
        )

        ctr_data = get_ctr_analytics()
        self.assertIn("total_searches", ctr_data)
        self.assertIn("total_clicks", ctr_data)
        self.assertIn("overall_ctr_pct", ctr_data)

    def test_api_v1_experiments_endpoint(self):
        res = self.client.get("/api/v1/experiments")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("experiments", data)
        self.assertGreater(len(data["experiments"]), 0)

    def test_api_v1_click_endpoint(self):
        res = self.client.post(
            "/api/v1/analytics/click",
            data=json.dumps({
                "request_id": "test_req_click",
                "doc_id": "python.txt",
                "position": 1,
                "search_method": "bm25"
            }),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "recorded")

    def test_api_v1_ctr_endpoint(self):
        res = self.client.get("/api/v1/analytics/ctr")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("overall_ctr_pct", data)


if __name__ == '__main__':
    unittest.main()
