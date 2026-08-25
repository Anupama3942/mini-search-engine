import unittest
import sys
import tempfile
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine, intersect_sorted_postings
from cache import BoundedLRUCache
import config


class TestOptimizationAndScalability(unittest.TestCase):
    
    def setUp(self):
        self.raw_documents = {
            "doc1.txt": "python programming language",
            "doc2.txt": "python is a programming language",
            "doc3.txt": "programming python language",
            "doc4.txt": "machine learning algorithm",
            "doc5.txt": "machine algorithm learning",
            "doc6.txt": "python programming is powerful. python programming is fun.",
            "doc7.txt": "java programming javascript"
        }
        self.engine = SearchEngine(documents=dict(self.raw_documents))

    def test_bounded_lru_cache(self):
        cache = BoundedLRUCache(maxsize=2, name="test_cache")
        cache.set("a", 1)
        cache.set("b", 2)
        
        self.assertEqual(cache.get("a"), 1) # Hit
        self.assertEqual(cache.hits, 1)
        self.assertIsNone(cache.get("c"))   # Miss
        self.assertEqual(cache.misses, 1)

        # Adding 'c' should evict 'b' (since 'a' was recently accessed)
        cache.set("c", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)

        stats = cache.get_stats()
        self.assertEqual(stats["size"], 2)
        self.assertEqual(stats["maxsize"], 2)
        self.assertGreater(stats["hit_rate_pct"], 0)

    def test_query_cache_hit_and_miss(self):
        query = "python AND programming"
        
        # 1st search -> Cache Miss
        res1 = self.engine.search(query, log_analytics=False)
        self.assertFalse(res1.cache_hit)
        
        # 2nd search -> Cache Hit
        res2 = self.engine.search(query, log_analytics=False)
        self.assertTrue(res2.cache_hit)
        
        # Verify results are identical
        self.assertEqual(len(res1), len(res2))
        self.assertEqual([r["filename"] for r in res1], [r["filename"] for r in res2])

    def test_cache_invalidation_on_rebuild(self):
        self.engine.search("python", log_analytics=False)
        self.assertEqual(self.engine.query_cache.get_stats()["size"], 1)
        
        # Rebuilding index must invalidate caches and bump version
        old_version = self.engine.index_version
        self.engine._build_index()
        self.assertEqual(self.engine.query_cache.get_stats()["size"], 0)
        self.assertGreater(self.engine.index_version, old_version)

    def test_incremental_indexing_add_and_remove(self):
        initial_doc_count = len(self.engine.documents)
        initial_vocab_size = len(self.engine.inverted_index)

        # 1. Add Document
        new_doc_name = "doc8.txt"
        new_content = "rust systems programming language"
        self.engine.add_document(new_doc_name, new_content)

        self.assertEqual(len(self.engine.documents), initial_doc_count + 1)
        self.assertIn("rust", self.engine.inverted_index)
        self.assertIn("doc8.txt", self.engine.inverted_index["rust"])

        # Search for newly added term
        results = self.engine.search("rust", log_analytics=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["filename"], "doc8.txt")

        # 2. Remove Document
        removed = self.engine.remove_document("doc8.txt")
        self.assertTrue(removed)
        self.assertEqual(len(self.engine.documents), initial_doc_count)
        self.assertNotIn("rust", self.engine.inverted_index)

        # Search again -> should return 0 results
        results_after = self.engine.search("rust", log_analytics=False)
        self.assertEqual(len(results_after), 0)

    def test_index_validation(self):
        report = self.engine.validate_index()
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["error_count"], 0)

    def test_two_pointer_posting_intersection(self):
        list_a = ["doc1.txt", "doc2.txt", "doc5.txt", "doc7.txt"]
        list_b = ["doc2.txt", "doc5.txt", "doc8.txt"]
        intersection = intersect_sorted_postings(list_a, list_b)
        self.assertEqual(intersection, ["doc2.txt", "doc5.txt"])

    def test_precomputed_tf_and_idf(self):
        # Verify TF lookup is O(1) and math matches
        # doc1.txt: "python programming language" -> 3 tokens
        tf_python = self.engine.calculate_tf("python", "doc1.txt")
        self.assertAlmostEqual(tf_python, 1.0 / 3.0)

        # Verify IDF is precalculated
        idf_python = self.engine.calculate_idf("python")
        self.assertGreater(idf_python, 0.0)

    def test_candidate_reduction_metric(self):
        res = self.engine.search("python AND java", log_analytics=False)
        self.assertIn("candidate_reduction_pct", dir(res))
        self.assertGreaterEqual(res.candidate_reduction_pct, 0.0)

    def test_health_check(self):
        health = self.engine.health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["index_valid"])
        self.assertTrue(health["index_loaded"])
        self.assertIn("memory_usage", health)

    def test_index_serialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_index_cache.json"
            
            # Save
            success_save = self.engine.save_index(save_path)
            self.assertTrue(success_save)
            self.assertTrue(save_path.exists())

            # Load into a new engine instance
            new_engine = SearchEngine(documents={})
            success_load = new_engine.load_index(save_path)
            self.assertTrue(success_load)
            self.assertEqual(len(new_engine.inverted_index), len(self.engine.inverted_index))
            self.assertEqual(new_engine.index_stats["total_documents"], self.engine.index_stats["total_documents"])

    def test_search_correctness_regression(self):
        # 1. Exact phrase
        r_phrase = self.engine.search('"python programming"', log_analytics=False)
        self.assertEqual({r['filename'] for r in r_phrase}, {"doc1.txt", "doc2.txt", "doc6.txt"})

        # 2. Boolean AND
        r_and = self.engine.search("python AND programming", log_analytics=False)
        self.assertEqual(len(r_and), 4)

        # 3. Boolean NOT
        r_not = self.engine.search("programming AND NOT java", log_analytics=False)
        self.assertNotIn("doc7.txt", {r['filename'] for r in r_not})

        # 4. Fuzzy search
        r_fuzzy = self.engine.search("pythn", log_analytics=False)
        self.assertEqual(r_fuzzy.did_you_mean, "python")
        self.assertGreater(len(r_fuzzy), 0)


if __name__ == '__main__':
    unittest.main()
