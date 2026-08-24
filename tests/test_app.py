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
        
    def test_search_empty_query(self):
        response = self.app.get('/search?q=')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Please enter a search term', response.data)

    def test_search_no_results(self):
        response = self.app.get('/search?q=blockchain')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'No results found', response.data)

    def test_404_page(self):
        response = self.app.get('/unknown-page')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Page Not Found', response.data)

    def test_xss_prevention(self):
        # Pass a script tag to ensure it's escaped
        # The script should not execute, it should be shown safely.
        malicious_query = "<script>alert(1)</script>"
        response = self.app.get(f'/search?q={malicious_query}')
        self.assertEqual(response.status_code, 200)
        
        # In Jinja, variables are autoescaped by default unless |safe is used.
        # Check if the query was safely escaped in the output
        self.assertNotIn(b'<script>alert(1)</script>', response.data)
        self.assertIn(b'&lt;script&gt;alert(1)&lt;/script&gt;', response.data)


if __name__ == '__main__':
    unittest.main()
