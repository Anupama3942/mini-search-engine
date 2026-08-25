import unittest
import sys
import json
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import config
from app import app
from services.search_service import SearchService
from services.index_manager import IndexManager
from services.metrics import metrics_registry, MetricsRegistry
from services.retrieval import BM25Retriever, SemanticRetriever, HybridRetriever


class TestProductionArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.service = SearchService.get_instance()
        cls.manager = IndexManager()

    def test_search_service_parameter_validation(self):
        # Empty query
        res1 = self.service.search(query="")
        self.assertIn("error", res1)
        self.assertEqual(res1["status_code"], 400)

        # Query exceeding MAX_QUERY_LENGTH
        long_q = "a" * (config.MAX_QUERY_LENGTH + 10)
        res2 = self.service.search(query=long_q)
        self.assertIn("error", res2)
        self.assertEqual(res2["status_code"], 400)

        # Invalid top_k
        res3 = self.service.search(query="python", top_k=0)
        self.assertIn("error", res3)
        self.assertEqual(res3["status_code"], 400)

        # Invalid page
        res4 = self.service.search(query="python", page=0)
        self.assertIn("error", res4)
        self.assertEqual(res4["status_code"], 400)

    def test_search_service_response_schema_and_pagination(self):
        resp = self.service.search(query="python", method="bm25", top_k=10, page=1, limit=3)
        self.assertIn("request_id", resp)
        self.assertEqual(resp["query"], "python")
        self.assertEqual(resp["method"], "bm25")
        self.assertGreater(resp["total_results"], 0)
        self.assertEqual(resp["page"], 1)
        self.assertEqual(resp["limit"], 3)
        self.assertIn("total_pages", resp)
        self.assertIn("results", resp)
        self.assertLessEqual(len(resp["results"]), 3)

    def test_two_stage_retrieval_pipelines(self):
        # Two-stage BM25 -> LTR
        res_bm25_ltr = self.service.search(query="python programming", method="bm25_ltr", top_k=5)
        self.assertNotIn("error", res_bm25_ltr)
        self.assertGreater(len(res_bm25_ltr["results"]), 0)

        # Two-stage Hybrid -> LTR
        res_hyb_ltr = self.service.search(query="web development", method="hybrid_ltr", top_k=5)
        self.assertNotIn("error", res_hyb_ltr)
        self.assertGreater(len(res_hyb_ltr["results"]), 0)

    def test_retrievers(self):
        bm25_ret = BM25Retriever(self.service.engine)
        candidates = bm25_ret.retrieve("python", top_k=5)
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)

        sem_ret = SemanticRetriever()
        sem_candidates = sem_ret.retrieve("python", top_k=5)
        self.assertIsInstance(sem_candidates, list)

        hyb_ret = HybridRetriever(self.service.engine)
        hyb_candidates = hyb_ret.retrieve("python", top_k=5)
        self.assertIsInstance(hyb_candidates, list)
        self.assertGreater(len(hyb_candidates), 0)

    def test_index_manager_readiness_and_validation(self):
        health = self.manager.get_health()
        self.assertEqual(health["status"], "ready")
        self.assertTrue(health["ready"])
        self.assertIn("validation", health)

    def test_metrics_registry(self):
        metrics_registry.record_request("bm25", 0.005, success=True)
        metrics_registry.record_cache_event(hit=True)
        metrics_registry.record_cache_event(hit=False)

        data = metrics_registry.to_dict()
        self.assertGreater(data["total_requests"], 0)
        self.assertIn("latency_stats", data)
        self.assertIn("p95_ms", data["latency_stats"])

        prom_text = metrics_registry.to_prometheus()
        self.assertIn("search_requests_total", prom_text)
        self.assertIn("search_latency_p95_ms", prom_text)

    def test_api_v1_search_endpoint(self):
        response = self.client.get("/api/v1/search?q=python&method=bm25&top_k=5&page=1&limit=5")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["query"], "python")
        self.assertIn("results", data)
        self.assertIn("X-Request-ID", response.headers)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_api_v1_health_and_ready_endpoints(self):
        res_h = self.client.get("/api/v1/health")
        self.assertEqual(res_h.status_code, 200)
        data_h = json.loads(res_h.data)
        self.assertEqual(data_h["status"], "healthy")

        res_r = self.client.get("/api/v1/ready")
        self.assertEqual(res_r.status_code, 200)
        data_r = json.loads(res_r.data)
        self.assertTrue(data_r["ready"])

    def test_api_v1_metrics_endpoint(self):
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("total_requests", data)
        self.assertIn("latency_stats", data)

    def test_prometheus_metrics_route(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("search_requests_total", response.data.decode("utf-8"))


if __name__ == '__main__':
    unittest.main()
