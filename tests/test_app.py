import unittest
import sys
from pathlib import Path

# Add the parent directory to sys.path to import app.py
sys.path.append(str(Path(__file__).parent.parent))

from app import app

class TestAppRoutes(unittest.TestCase):
    
    def setUp(self):
        # Create a test client
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MINI SEARCH ENGINE', response.data)

    def test_search_valid_query(self):
        response = self.app.get('/search?q=python')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Results found', response.data)

    def test_search_fuzzy_query(self):
        # Searching for typo 'pythn'
        response = self.app.get('/search?q=pythn')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Did you mean:', response.data)
        self.assertIn(b'python', response.data)

    def test_search_fuzzy_boolean(self):
        response = self.app.get('/search?q=pythn+AND+programming')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'python AND programming', response.data)

    def test_search_boolean_query(self):
        response = self.app.get('/search?q=python+AND+programming')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Results found', response.data)

    def test_search_phrase_query(self):
        response = self.app.get('/search?q="python+programming"')
        self.assertEqual(response.status_code, 200)

    def test_search_empty_query(self):
        response = self.app.get('/search?q=')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Please enter a search term', response.data)

    def test_search_no_results(self):
        response = self.app.get('/search?q=qwertyuiopasdf')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'No results found', response.data)

    def test_404_page(self):
        response = self.app.get('/unknown-page')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Page Not Found', response.data)

    def test_xss_prevention_plain(self):
        malicious_query = "<script>alert('xss')</script>"
        response = self.app.get(f'/search?q={malicious_query}')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'<script>alert(\'xss\')</script>', response.data)
        self.assertIn(b'&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;', response.data)

    def test_xss_prevention_fuzzy(self):
        malicious_query = "pythn<script>alert('xss')</script>"
        response = self.app.get(f'/search?q={malicious_query}')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'<script>alert(\'xss\')</script>', response.data)

    # --- Stage 10/11 Analytics Dashboard, Health & API Tests ---
    def test_analytics_dashboard_route(self):
        self.app.get('/search?q=python')
        response = self.app.get('/analytics')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Search Analytics & Performance', response.data)
        self.assertIn(b'Performance Optimization & Caching', response.data)

    def test_health_check_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["index_valid"])

    def test_api_analytics_cache(self):
        response = self.app.get('/api/analytics/cache')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("query_cache", data)
        self.assertIn("fuzzy_cache", data)

    def test_api_analytics_summary(self):
        response = self.app.get('/api/analytics/summary')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('total_searches', data)
        self.assertIn('avg_latency_ms', data)

    def test_api_analytics_top_queries(self):
        response = self.app.get('/api/analytics/top-queries')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)

    def test_api_analytics_performance(self):
        response = self.app.get('/api/analytics/performance')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('latency_percentiles_ms', data)
        self.assertIn('memory_allocation', data)

    def test_api_analytics_index(self):
        response = self.app.get('/api/analytics/index')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('total_documents', data)
        self.assertIn('vocabulary_size', data)


if __name__ == '__main__':
    unittest.main()
