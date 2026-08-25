"""
Mini Search Engine - Stage 14
LTR Dataset Construction & Query-Level Split
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from query_parser import tokenize_query, QueryParser, extract_positive_terms, resolve_ast
from .features import FeatureExtractor, FEATURE_NAMES
from evaluation.evaluator import DEFAULT_JUDGMENTS_PATH


@dataclass
class QuerySample:
    query_id: str
    query_text: str
    doc_ids: List[str]
    X: List[List[float]]
    y: List[float]


class LTRDatasetBuilder:
    """Builds query-grouped feature matrices from ground truth judgments."""

    def __init__(self, extractor: Optional[FeatureExtractor] = None):
        self.extractor = extractor or FeatureExtractor()

    def build_dataset(
        self, 
        engine, 
        dataset_path: Path = DEFAULT_JUDGMENTS_PATH
    ) -> List[QuerySample]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = data.get("queries", [])
        all_docs = list(engine.documents.keys())
        vocabulary = set(engine.inverted_index.keys())
        from search import process_text

        samples = []

        for q_item in queries:
            qid = q_item["query_id"]
            query_str = q_item["query"]
            relevant_docs = set(q_item.get("relevant_documents", []))

            # Process query
            try:
                tokens = tokenize_query(query_str, process_text)
                parser = QueryParser(tokens)
                ast = parser.parse()
                resolved_ast, corrections = resolve_ast(ast, vocabulary, engine.fuzzy_cache)
                positive_terms = list(set(extract_positive_terms(resolved_ast)))
            except Exception:
                positive_terms = query_str.lower().split()
                corrections = {}

            # Generate features for all corpus documents for this query
            doc_ids = []
            X_rows = []
            y_rows = []

            for doc_id in all_docs:
                vec = self.extractor.extract_vector(
                    positive_terms, 
                    doc_id, 
                    engine, 
                    raw_query=query_str, 
                    fuzzy_corrections=corrections
                )
                label = 1.0 if doc_id in relevant_docs else 0.0

                doc_ids.append(doc_id)
                X_rows.append(vec)
                y_rows.append(label)

            samples.append(QuerySample(
                query_id=qid,
                query_text=query_str,
                doc_ids=doc_ids,
                X=X_rows,
                y=y_rows
            ))

        return samples

    def split_queries(
        self, 
        samples: List[QuerySample], 
        train_ratio: float = 0.70, 
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[QuerySample], List[QuerySample], List[QuerySample]]:
        """
        Split dataset at the query level to prevent information leakage.
        """
        shuffled = list(samples)
        random.seed(seed)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(int(n * train_ratio), 1)
        n_val = max(int(n * val_ratio), 1)

        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train:n_train + n_val]
        test_samples = shuffled[n_train + n_val:]

        # Ensure test has at least 1 query if dataset is small
        if not test_samples and len(val_samples) > 1:
            test_samples.append(val_samples.pop())

        return train_samples, val_samples, test_samples

    def flatten_dataset(self, samples: List[QuerySample]) -> Tuple[List[List[float]], List[float]]:
        """Flatten query-grouped samples into global (X, y) matrices."""
        X_all = []
        y_all = []
        for s in samples:
            X_all.extend(s.X)
            y_all.extend(s.y)
        return X_all, y_all

    def generate_pairwise_differences(self, samples: List[QuerySample]) -> List[List[float]]:
        """
        Generate feature differences (x_rel - x_nonrel) for pairwise ranking.
        """
        diffs = []
        for s in samples:
            rel_indices = [i for i, label in enumerate(s.y) if label == 1.0]
            nonrel_indices = [i for i, label in enumerate(s.y) if label == 0.0]

            for r_idx in rel_indices:
                x_rel = s.X[r_idx]
                for nr_idx in nonrel_indices:
                    x_nr = s.X[nr_idx]
                    diff = [r_val - nr_val for r_val, nr_val in zip(x_rel, x_nr)]
                    diffs.append(diff)
        return diffs
