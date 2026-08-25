import unittest
import sys
import math
import tempfile
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from semantic.embeddings import (
    normalize_l2,
    DenseSemanticEmbeddingModel,
    EmbeddingService
)
from semantic.vector_store import NumpyVectorStore, cosine_similarity
from semantic.hybrid import HybridSearchEngine, min_max_normalize
from ranking.semantic import SemanticRanker
from ranking.hybrid import HybridRanker
import config


class TestSemanticSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = SearchEngine()
        cls.service = EmbeddingService.get_instance()

    def test_vector_normalization(self):
        v = [3.0, 4.0]
        norm_v = normalize_l2(v)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in norm_v)), 1.0, places=5)
        self.assertEqual(norm_v, [0.6, 0.8])

        # Zero vector safety
        zero_v = [0.0, 0.0, 0.0]
        self.assertEqual(normalize_l2(zero_v), [0.0, 0.0, 0.0])

    def test_cosine_similarity_edge_cases(self):
        # Identical vectors
        v1 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v1), 1.0, places=5)

        # Orthogonal vectors
        v2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0, places=5)

        # Opposite vectors
        v3 = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v3), -1.0, places=5)

        # Zero vector handling (must not divide by zero)
        v_zero = [0.0, 0.0, 0.0]
        self.assertEqual(cosine_similarity(v1, v_zero), 0.0)

        # Dimension mismatch
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_embedding_service_caching_and_batching(self):
        text = "Python programming language"
        vec1 = self.service.encode(text)
        self.assertEqual(len(vec1), self.service.dimension)

        # Second call should hit cache
        vec2 = self.service.encode(text)
        self.assertEqual(vec1, vec2)

        # Batch encode
        texts = ["Text A", "Text B", "Text C"]
        batch_vecs = self.service.encode_batch(texts)
        self.assertEqual(len(batch_vecs), 3)
        self.assertEqual(len(batch_vecs[0]), self.service.dimension)

    def test_numpy_vector_store_crud_and_search(self):
        store = NumpyVectorStore(dimension=3)
        self.assertEqual(store.search([1.0, 0.0, 0.0]), [])

        store.add_document("doc1.txt", [1.0, 0.0, 0.0])
        store.add_document("doc2.txt", [0.0, 1.0, 0.0])
        store.add_document("doc3.txt", [0.8, 0.6, 0.0])

        self.assertEqual(len(store.vectors), 3)

        # Search for query vector closest to doc1 and doc3
        results = store.search([1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["filename"], "doc1.txt")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=4)
        self.assertEqual(results[1]["filename"], "doc3.txt")

        # Remove document
        self.assertTrue(store.remove_document("doc2.txt"))
        self.assertEqual(len(store.vectors), 2)
        self.assertFalse(store.remove_document("nonexistent.txt"))

    def test_vector_store_save_and_load(self):
        store = NumpyVectorStore(dimension=4)
        store.add_document("sample.txt", [0.5, 0.5, 0.5, 0.5])

        with tempfile.TemporaryDirectory() as tmp_dir:
            idx_file = Path(tmp_dir) / "idx.json"
            meta_file = Path(tmp_dir) / "meta.json"

            self.assertTrue(store.save(idx_file, meta_file))
            self.assertTrue(idx_file.exists())
            self.assertTrue(meta_file.exists())

            loaded_store = NumpyVectorStore(dimension=4)
            self.assertTrue(loaded_store.load(idx_file, meta_file))
            self.assertEqual(len(loaded_store.vectors), 1)
            self.assertIn("sample.txt", loaded_store.vectors)

    def test_semantic_ranker_bm25_fallback(self):
        empty_store = NumpyVectorStore(dimension=64)
        ranker = SemanticRanker(vector_store=empty_store)
        self.assertFalse(ranker.is_available)

        # Should fall back to BM25 scoring gracefully
        score = ranker.score(["python"], "python.txt", self.engine)
        self.assertGreater(score, 0.0)

    def test_hybrid_min_max_score_normalization(self):
        scores = {"d1": 10.0, "d2": 20.0, "d3": 30.0}
        norm = min_max_normalize(scores)
        self.assertEqual(norm["d1"], 0.0)
        self.assertEqual(norm["d2"], 0.5)
        self.assertEqual(norm["d3"], 1.0)

        # Single score / zero variance
        single = {"d1": 15.0}
        self.assertEqual(min_max_normalize(single)["d1"], 1.0)

        empty = {}
        self.assertEqual(min_max_normalize(empty), {})

    def test_hybrid_search_scoring_and_alpha_bounds(self):
        hybrid_engine = HybridSearchEngine()
        results = hybrid_engine.search_hybrid("python programming", self.engine, alpha=0.5, top_k=5)
        self.assertGreater(len(results), 0)
        
        # Test alpha bounds validation
        with self.assertRaises(ValueError):
            config.validate_hybrid_params(-0.1)
        with self.assertRaises(ValueError):
            config.validate_hybrid_params(1.5)

    def test_full_search_engine_semantic_and_hybrid_query(self):
        res_sem = self.engine.search("python programming", ranking_algorithm="semantic", top_k=5)
        self.assertGreater(len(res_sem), 0)
        self.assertEqual(res_sem.ranking_algorithm, "semantic")

        res_hyb = self.engine.search("python programming", ranking_algorithm="hybrid", top_k=5)
        self.assertGreater(len(res_hyb), 0)
        self.assertEqual(res_hyb.ranking_algorithm, "hybrid")


if __name__ == '__main__':
    unittest.main()
