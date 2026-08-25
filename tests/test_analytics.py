import unittest
import sys
import tempfile
import gc
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from analytics import (
    init_db,
    record_search,
    get_summary_metrics,
    get_top_queries,
    get_top_zero_result_queries,
    get_query_type_distribution,
    get_recent_searches
)
from performance import calculate_percentiles, get_memory_usage


class TestAnalyticsAndPerformance(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_analytics.db"
        init_db(self.db_path)

    def tearDown(self):
        gc.collect() # Ensure all sqlite connections are collected
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_percentile_calculation_known_values(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        pct = calculate_percentiles(vals)
        self.assertEqual(pct["count"], 5)
        self.assertEqual(pct["min"], 1.0)
        self.assertEqual(pct["max"], 5.0)
        self.assertEqual(pct["avg"], 3.0)
        self.assertEqual(pct["p50"], 3.0) # Median
        self.assertEqual(pct["p95"], 5.0) # 95th percentile

    def test_percentile_calculation_empty(self):
        pct = calculate_percentiles([])
        self.assertEqual(pct["count"], 0)
        self.assertEqual(pct["avg"], 0.0)

    def test_memory_usage_structure(self):
        mem = get_memory_usage()
        self.assertIn("current_kb", mem)
        self.assertIn("peak_kb", mem)
        self.assertIn("current_mb", mem)
        self.assertIn("peak_mb", mem)
        self.assertGreaterEqual(mem["current_bytes"], 0)

    def test_record_search_and_summary_metrics(self):
        # Record normal search
        record_search("python", 5, 0.010, query_type="normal", db_path=self.db_path)
        # Record zero-result search
        record_search("quantum_banana", 0, 0.005, query_type="normal", db_path=self.db_path)
        # Record fuzzy search
        record_search("pythn", 4, 0.020, query_type="fuzzy", fuzzy_used=True, db_path=self.db_path)
        # Record boolean search
        record_search("python AND programming", 3, 0.015, query_type="boolean", boolean_used=True, db_path=self.db_path)

        summary = get_summary_metrics(self.db_path)
        self.assertEqual(summary["total_searches"], 4)
        self.assertEqual(summary["zero_result_count"], 1)
        self.assertEqual(summary["zero_result_rate"], 25.0) # 1 / 4 = 25%
        self.assertEqual(summary["fuzzy_usage_count"], 1)
        self.assertEqual(summary["fuzzy_usage_rate"], 25.0)
        self.assertEqual(summary["boolean_usage_count"], 1)
        self.assertEqual(summary["boolean_usage_rate"], 25.0)
        self.assertGreater(summary["avg_latency_ms"], 0.0)

    def test_top_queries_and_zero_result_queries(self):
        record_search("python", 3, 0.010, db_path=self.db_path)
        record_search("python", 3, 0.012, db_path=self.db_path)
        record_search("database", 2, 0.008, db_path=self.db_path)
        record_search("notfound1", 0, 0.004, db_path=self.db_path)
        record_search("notfound1", 0, 0.005, db_path=self.db_path)

        top_q = get_top_queries(limit=5, db_path=self.db_path)
        self.assertEqual(len(top_q), 3)
        self.assertEqual(top_q[0]["query"], "notfound1")
        self.assertEqual(top_q[0]["count"], 2)
        self.assertEqual(top_q[1]["query"], "python")
        self.assertEqual(top_q[1]["count"], 2)

        zero_q = get_top_zero_result_queries(limit=5, db_path=self.db_path)
        self.assertEqual(len(zero_q), 1)
        self.assertEqual(zero_q[0]["query"], "notfound1")
        self.assertEqual(zero_q[0]["count"], 2)

    def test_query_type_distribution(self):
        record_search("python", 2, 0.010, db_path=self.db_path)
        record_search('"machine learning"', 1, 0.015, phrase_used=True, query_type="phrase", db_path=self.db_path)

        dist = get_query_type_distribution(self.db_path)
        self.assertEqual(dist["total"], 2)
        self.assertEqual(dist["normal"]["count"], 1)
        self.assertEqual(dist["phrase"]["count"], 1)

    def test_recent_searches(self):
        record_search("first", 1, 0.01, db_path=self.db_path)
        record_search("second", 2, 0.02, db_path=self.db_path)

        recent = get_recent_searches(limit=10, db_path=self.db_path)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["query"], "second")
        self.assertEqual(recent[1]["query"], "first")

    def test_search_engine_index_statistics(self):
        engine = SearchEngine()
        stats = engine.get_index_statistics()
        self.assertIn("total_documents", stats)
        self.assertIn("vocabulary_size", stats)
        self.assertIn("total_tokens", stats)
        self.assertIn("total_postings", stats)
        self.assertIn("build_time_seconds", stats)
        self.assertIn("throughput_docs_per_sec", stats)
        self.assertGreater(stats["total_documents"], 0)
        self.assertGreater(stats["vocabulary_size"], 0)

    def test_search_engine_timing_instrumentation(self):
        engine = SearchEngine()
        results = engine.search("python", log_analytics=False)
        self.assertIn("query_parsing_time", results.timings)
        self.assertIn("term_resolution_time", results.timings)
        self.assertIn("retrieval_time", results.timings)
        self.assertIn("ranking_time", results.timings)
        self.assertIn("total_search_duration", results.timings)
        self.assertGreater(results.timings["total_search_duration"], 0)

    def test_fault_tolerance_analytics_failure_does_not_break_search(self):
        # Point analytics to an invalid file path where table cannot be written
        invalid_path = Path(self.temp_dir.name) / "nonexistent_dir" / "invalid.db"
        
        # Test recording fails gracefully
        success = record_search("test", 1, 0.01, db_path=invalid_path)
        self.assertFalse(success)

        # Ensure search still executes without crashing even if analytics throws
        engine = SearchEngine()
        results = engine.search("python", log_analytics=True)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_sql_injection_protection(self):
        malicious_query = "'; DROP TABLE search_events; --"
        # Should record safely as literal text without dropping table
        success = record_search(malicious_query, 0, 0.01, db_path=self.db_path)
        self.assertTrue(success)

        # Verify table still exists and query was recorded
        summary = get_summary_metrics(self.db_path)
        self.assertEqual(summary["total_searches"], 1)
        top = get_top_queries(limit=1, db_path=self.db_path)
        self.assertEqual(top[0]["query"], malicious_query.strip().lower())


if __name__ == '__main__':
    unittest.main()
