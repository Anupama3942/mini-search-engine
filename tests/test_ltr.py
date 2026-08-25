import unittest
import sys
import math
import tempfile
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from search import SearchEngine
from ltr.features import FeatureExtractor, FeatureScaler, FEATURE_NAMES, FEATURE_VERSION
from ltr.models import PointwiseLogisticRegressionModel, PairwiseRankerModel
from ltr.dataset import LTRDatasetBuilder, QuerySample
from ltr.ablation import FeatureAblationExperiment
from ranking.ltr import LTRRanker
from evaluation.metrics import (
    discounted_cumulative_gain,
    ideal_discounted_cumulative_gain,
    ndcg_at_k,
    mean_ndcg_at_k
)
import config


class TestLearningToRank(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = SearchEngine()
        cls.extractor = FeatureExtractor()

    def test_feature_ordering_and_names(self):
        self.assertEqual(len(FEATURE_NAMES), 10)
        self.assertIn("bm25_score", FEATURE_NAMES)
        self.assertIn("tfidf_score", FEATURE_NAMES)
        self.assertIn("query_term_coverage", FEATURE_NAMES)
        self.assertIn("phrase_match", FEATURE_NAMES)
        self.assertIn("title_match", FEATURE_NAMES)
        self.assertIn("fuzzy_score", FEATURE_NAMES)

    def test_feature_extraction_consistency(self):
        # The same (query, document) pair must generate the exact same feature vector
        vec1 = self.extractor.extract_vector(["python", "programming"], "python.txt", self.engine)
        vec2 = self.extractor.extract_vector(["python", "programming"], "python.txt", self.engine)
        
        self.assertEqual(len(vec1), 10)
        self.assertEqual(vec1, vec2)

        named = self.extractor.extract_named_features(["python"], "python.txt", self.engine)
        self.assertGreater(named["bm25_score"], 0.0)
        self.assertGreater(named["tfidf_score"], 0.0)
        self.assertEqual(named["query_term_coverage"], 1.0)
        self.assertEqual(named["title_match"], 1.0)

    def test_feature_scaler(self):
        scaler = FeatureScaler()
        X = [
            [0.0, 10.0, 100.0],
            [1.0, 20.0, 200.0],
            [2.0, 30.0, 300.0]
        ]
        scaler.fit(X)
        transformed = scaler.transform(X)

        self.assertEqual(transformed[0], [0.0, 0.0, 0.0])
        self.assertEqual(transformed[1], [0.5, 0.5, 0.5])
        self.assertEqual(transformed[2], [1.0, 1.0, 1.0])

    def test_pointwise_logistic_regression_fit_and_predict(self):
        model = PointwiseLogisticRegressionModel(epochs=500, regularization_c=1.0)
        # Linearly separable 2-feature toy data
        X = [[1.0, 1.0], [1.2, 0.9], [0.1, 0.2], [0.0, 0.1]]
        y = [1.0, 1.0, 0.0, 0.0]

        model.fit(X, y, feature_names=["f1", "f2"])
        self.assertTrue(model.is_trained)

        # Positive query should score high
        prob_pos = model.predict_proba_vector([1.1, 1.0])
        prob_neg = model.predict_proba_vector([0.05, 0.05])
        self.assertGreater(prob_pos, prob_neg)
        self.assertGreater(prob_pos, 0.5)
        self.assertLess(prob_neg, 0.5)

    def test_model_save_and_load(self):
        model = PointwiseLogisticRegressionModel(epochs=200)
        X = [[0.8, 0.5], [0.1, 0.2]]
        y = [1.0, 0.0]
        model.fit(X, y, feature_names=["f1", "f2"])

        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "model.json"
            meta_file = Path(temp_dir) / "meta.json"

            self.assertTrue(model.save(model_file, meta_file))
            self.assertTrue(model_file.exists())
            self.assertTrue(meta_file.exists())

            loaded_model = PointwiseLogisticRegressionModel()
            self.assertTrue(loaded_model.load(model_file))
            self.assertEqual(loaded_model.feature_names, ["f1", "f2"])
            self.assertEqual(loaded_model.weights, model.weights)
            self.assertAlmostEqual(
                loaded_model.predict_proba_vector([0.8, 0.5]),
                model.predict_proba_vector([0.8, 0.5]),
                places=5
            )

    def test_feature_version_mismatch_fails_load(self):
        model = PointwiseLogisticRegressionModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "incompatible_model.json"
            meta_file = Path(temp_dir) / "meta.json"
            
            model.feature_version = "99.0"
            model.save(model_file, meta_file)

            new_model = PointwiseLogisticRegressionModel()
            self.assertFalse(new_model.load(model_file))

    def test_ltr_ranker_bm25_fallback_when_model_missing(self):
        fake_path = Path("non_existent_dir_123/fake_model.json")
        ranker = LTRRanker(model_path=fake_path)
        self.assertFalse(ranker.is_ready)

        # Fallback to BM25 scoring should succeed without error
        score = ranker.score(["python"], "python.txt", self.engine)
        self.assertGreater(score, 0.0)

    def test_ndcg_at_k_hand_calculation(self):
        # Perfect ranking: [rel, rel, non]
        # DCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        # IDCG@3 = 1.6309
        # NDCG@3 = 1.0
        retrieved_perfect = ["d1", "d2", "d3"]
        relevant = ["d1", "d2"]
        self.assertAlmostEqual(ndcg_at_k(retrieved_perfect, relevant, k=3), 1.0, places=3)

        # Inverted ranking: [non, rel, rel]
        # DCG@3 = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0.6309 + 0.5 = 1.1309
        # IDCG@3 = 1.6309
        # NDCG@3 = 1.1309 / 1.6309 ≈ 0.6934
        retrieved_inverted = ["d3", "d1", "d2"]
        expected_ndcg = round(1.13093 / 1.63093, 4)
        self.assertAlmostEqual(ndcg_at_k(retrieved_inverted, relevant, k=3), expected_ndcg, places=3)

        # Empty retrieved or relevant
        self.assertEqual(ndcg_at_k([], []), 1.0)
        self.assertEqual(ndcg_at_k([], ["d1"]), 0.0)

    def test_query_grouped_split_zero_leakage(self):
        builder = LTRDatasetBuilder(self.extractor)
        samples = builder.build_dataset(self.engine)
        train_s, val_s, test_s = builder.split_queries(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        train_qids = set(s.query_id for s in train_s)
        val_qids = set(s.query_id for s in val_s)
        test_qids = set(s.query_id for s in test_s)

        # Assert no overlap
        self.assertEqual(len(train_qids & val_qids), 0)
        self.assertEqual(len(train_qids & test_qids), 0)
        self.assertEqual(len(val_qids & test_qids), 0)

    def test_pairwise_model(self):
        model = PairwiseRankerModel(epochs=200)
        # Toy preference differences (x_rel - x_nonrel)
        diffs = [[1.0, 0.5], [0.8, 0.6], [0.9, 0.4]]
        model.fit_pairs(diffs, feature_names=["f1", "f2"])
        self.assertTrue(model.is_trained)

        # High feature document should score higher than low feature document
        score_high = model.predict_score([2.0, 2.0])
        score_low = model.predict_score([0.1, 0.1])
        self.assertGreater(score_high, score_low)

    def test_feature_ablation_experiment(self):
        builder = LTRDatasetBuilder(self.extractor)
        samples = builder.build_dataset(self.engine)
        train_s, _, test_s = builder.split_queries(samples, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)

        ablation = FeatureAblationExperiment(builder)
        results = ablation.run_experiment(train_s, test_s)

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIn("feature_set", r)
            self.assertIn("map", r)
            self.assertIn("ndcg@5", r)
            self.assertGreaterEqual(r["map"], 0.70)


if __name__ == '__main__':
    unittest.main()
