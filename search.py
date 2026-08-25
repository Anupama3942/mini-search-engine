"""
Mini Search Engine - Stage 11 (Core Logic & Optimizations)
Performance, Caching, Precomputation, Incremental Indexing & Scalability.
"""

from pathlib import Path
import string
import math
import html
import re
import time
import json
import heapq
from typing import Dict, List, Any, Optional, Set, Tuple

import config
from cache import BoundedLRUCache
from query_parser import (
    tokenize_query, 
    QueryParser, 
    evaluate_query, 
    extract_positive_terms,
    resolve_ast,
    PhraseNode,
    AndNode,
    OrNode,
    NotNode
)
from analytics import record_search
from performance import get_memory_usage

# Predefined stop words
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "of",
    "in", "on", "to", "for", "with", "as", "at", "by", "from", "this",
    "that", "it"
}

def load_documents(documents_dir: Path = config.DOCUMENTS_DIR) -> Dict[str, str]:
    if not documents_dir.exists() or not documents_dir.is_dir():
        return {}

    txt_files = sorted(documents_dir.glob("*.txt"))
    documents = {}
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                documents[file_path.name] = file.read()
        except Exception:
            pass
    return documents

def normalize_text(text: str) -> str:
    text = text.lower()
    for punc in string.punctuation:
        text = text.replace(punc, " ")
    return text

def tokenize_text(text: str) -> List[str]:
    return text.split()

def remove_stop_words(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS]

def process_text(text: str) -> List[str]:
    normalized = normalize_text(text)
    tokens = tokenize_text(normalized)
    return remove_stop_words(tokens)

def generate_snippet(document_text: str, query_terms: List[str]) -> str:
    """Generate a short snippet and safely highlight query terms."""
    safe_text = html.escape(document_text)
    snippet = safe_text[:200]
    if len(safe_text) > 200:
        snippet += "..."
        
    for term in query_terms:
        safe_term = html.escape(term)
        pattern = re.compile(rf"\b{re.escape(safe_term)}\b", re.IGNORECASE)
        snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)

    return snippet

def intersect_sorted_postings(list_a: List[str], list_b: List[str]) -> List[str]:
    """
    Two-pointer intersection algorithm for sorted posting lists.
    Time Complexity: O(len(a) + len(b))
    """
    i, j = 0, 0
    intersection = []
    len_a, len_b = len(list_a), len(list_b)
    while i < len_a and j < len_b:
        if list_a[i] == list_b[j]:
            intersection.append(list_a[i])
            i += 1
            j += 1
        elif list_a[i] < list_b[j]:
            i += 1
        else:
            j += 1
    return intersection


class SearchResults(list):
    """List subclass holding search result items along with query metadata and performance timings."""
    def __init__(
        self, 
        items=None, 
        original_query="", 
        did_you_mean=None, 
        corrections=None,
        timings=None,
        query_type="normal",
        fuzzy_used=False,
        phrase_used=False,
        boolean_used=False,
        cache_hit=False,
        candidate_reduction_pct=0.0
    ):
        super().__init__(items or [])
        self.original_query = original_query
        self.did_you_mean = did_you_mean
        self.corrections = corrections or {}
        self.timings = timings or {}
        self.query_type = query_type
        self.fuzzy_used = fuzzy_used
        self.phrase_used = phrase_used
        self.boolean_used = boolean_used
        self.cache_hit = cache_hit
        self.candidate_reduction_pct = candidate_reduction_pct


def detect_ast_features(node) -> Tuple[bool, bool]:
    """Recursively inspect an AST node to detect Boolean and Phrase features."""
    if node is None:
        return False, False
    has_phrase = isinstance(node, PhraseNode)
    has_boolean = isinstance(node, (AndNode, OrNode, NotNode))
    
    if isinstance(node, (AndNode, OrNode)):
        l_p, l_b = detect_ast_features(node.left)
        r_p, r_b = detect_ast_features(node.right)
        return (has_phrase or l_p or r_p), (has_boolean or l_b or r_b)
    elif isinstance(node, NotNode):
        c_p, c_b = detect_ast_features(node.child)
        return (has_phrase or c_p), (has_boolean or c_b)
    
    return has_phrase, has_boolean


class SearchEngine:
    def __init__(self, documents: Optional[Dict[str, str]] = None):
        self.documents = documents if documents is not None else load_documents()
        self.processed_documents: Dict[str, List[str]] = {}
        self.inverted_index: Dict[str, Set[str]] = {}
        self.positional_index: Dict[str, Dict[str, List[int]]] = {}
        
        # Precomputed Document & Term Statistics (Stage 11 Optimization)
        self.doc_lengths: Dict[str, int] = {}
        self.term_counts: Dict[str, Dict[str, int]] = {}
        self.doc_freq: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}
        
        # Caching & Invalidation Layer (Stage 11 Optimization)
        self.index_version = 1
        self.query_cache = BoundedLRUCache(maxsize=config.QUERY_CACHE_SIZE, name="query_cache")
        self.fuzzy_cache = BoundedLRUCache(maxsize=config.FUZZY_CACHE_SIZE, name="fuzzy_cache")
        
        self.index_stats: Dict[str, Any] = {}
        
        if self.documents:
            self._build_index()

    def clear_caches(self) -> None:
        """Invalidate all dependent caches when the index or documents change."""
        self.index_version += 1
        self.query_cache.clear()
        self.fuzzy_cache.clear()
        self.idf_cache.clear()

    def _build_index(self) -> None:
        start_time = time.perf_counter()
        
        # Clear indexes, caches, and precomputed metadata
        self.inverted_index.clear()
        self.positional_index.clear()
        self.processed_documents.clear()
        self.doc_lengths.clear()
        self.term_counts.clear()
        self.doc_freq.clear()
        self.clear_caches()

        total_tokens = 0

        # 1. Process documents and precompute counts
        for filename, content in self.documents.items():
            tokens = process_text(content)
            self.processed_documents[filename] = tokens
            doc_len = len(tokens)
            self.doc_lengths[filename] = doc_len
            total_tokens += doc_len
            
            # Precompute term frequency counts per document (O(1) search TF lookup)
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.term_counts[filename] = counts

        # 2. Build inverted index and positional index
        total_postings = 0
        total_stored_positions = 0

        for filename, tokens in self.processed_documents.items():
            for position, token in enumerate(tokens):
                # Standard Inverted Index (Set for O(1) membership & union)
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(filename)
                
                # Positional Index: { term: { filename: [pos1, pos2] } }
                if token not in self.positional_index:
                    self.positional_index[token] = {}
                if filename not in self.positional_index[token]:
                    self.positional_index[token][filename] = []
                self.positional_index[token][filename].append(position)
                total_stored_positions += 1

        # 3. Precompute Document Frequency & IDF Cache for all terms
        doc_count = len(self.documents)
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.doc_freq[term] = df
            total_postings += df
            if doc_count > 0 and df > 0:
                self.idf_cache[term] = math.log(doc_count / df)
            else:
                self.idf_cache[term] = 0.0

        build_time = max(time.perf_counter() - start_time, 0.000001)
        vocab_size = len(self.inverted_index)

        largest_doc = max(self.doc_lengths.items(), key=lambda x: x[1]) if self.doc_lengths else ("None", 0)
        smallest_doc = min(self.doc_lengths.items(), key=lambda x: x[1]) if self.doc_lengths else ("None", 0)

        self.index_stats = {
            "total_documents": doc_count,
            "vocabulary_size": vocab_size,
            "total_tokens": total_tokens,
            "total_postings": total_postings,
            "total_stored_positions": total_stored_positions,
            "avg_postings_per_term": round(total_postings / vocab_size, 2) if vocab_size > 0 else 0.0,
            "avg_document_length": round(total_tokens / doc_count, 1) if doc_count > 0 else 0.0,
            "largest_document": {"filename": largest_doc[0], "tokens": largest_doc[1]},
            "smallest_document": {"filename": smallest_doc[0], "tokens": smallest_doc[1]},
            "build_time_seconds": round(build_time, 5),
            "throughput_docs_per_sec": round(doc_count / build_time, 2),
            "index_version": self.index_version
        }

    # --- Incremental Indexing (Stage 11 Optimization) ---
    def add_document(self, filename: str, content: str) -> None:
        """Incrementally add a document and update index structures without rebuilding the entire corpus."""
        self.documents[filename] = content
        tokens = process_text(content)
        self.processed_documents[filename] = tokens
        self.doc_lengths[filename] = len(tokens)
        
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        self.term_counts[filename] = counts

        for position, token in enumerate(tokens):
            if token not in self.inverted_index:
                self.inverted_index[token] = set()
            self.inverted_index[token].add(filename)

            if token not in self.positional_index:
                self.positional_index[token] = {}
            if filename not in self.positional_index[token]:
                self.positional_index[token][filename] = []
            self.positional_index[token][filename].append(position)

        # Invalidate caches & recompute IDFs
        self._recompute_stats()

    def remove_document(self, filename: str) -> bool:
        """Incrementally remove a document and clean inverted/positional index entries."""
        if filename not in self.documents:
            return False

        del self.documents[filename]
        if filename in self.processed_documents:
            del self.processed_documents[filename]
        if filename in self.doc_lengths:
            del self.doc_lengths[filename]
        if filename in self.term_counts:
            del self.term_counts[filename]

        # Clean inverted index
        terms_to_delete = []
        for term, postings in self.inverted_index.items():
            postings.discard(filename)
            if not postings:
                terms_to_delete.append(term)
        for term in terms_to_delete:
            del self.inverted_index[term]

        # Clean positional index
        pos_terms_to_delete = []
        for term, doc_dict in self.positional_index.items():
            doc_dict.pop(filename, None)
            if not doc_dict:
                pos_terms_to_delete.append(term)
        for term in pos_terms_to_delete:
            del self.positional_index[term]

        self._recompute_stats()
        return True

    def _recompute_stats(self) -> None:
        """Fast incremental statistic and IDF refresh."""
        self.clear_caches()
        doc_count = len(self.documents)
        self.doc_freq.clear()
        total_postings = 0
        
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.doc_freq[term] = df
            total_postings += df
            self.idf_cache[term] = math.log(doc_count / df) if doc_count > 0 and df > 0 else 0.0

        total_tokens = sum(self.doc_lengths.values())
        vocab_size = len(self.inverted_index)

        self.index_stats.update({
            "total_documents": doc_count,
            "vocabulary_size": vocab_size,
            "total_tokens": total_tokens,
            "total_postings": total_postings,
            "index_version": self.index_version
        })

    # --- Index Validation (Stage 11 Optimization) ---
    def validate_index(self) -> Dict[str, Any]:
        """Perform comprehensive integrity validation on index data structures."""
        errors = []
        doc_keys = set(self.documents.keys())

        # Check inverted index postings
        for term, postings in self.inverted_index.items():
            for doc in postings:
                if doc not in doc_keys:
                    errors.append(f"Inverted index term '{term}' references unknown document '{doc}'.")

        # Check positional index
        for term, doc_dict in self.positional_index.items():
            for doc, positions in doc_dict.items():
                if doc not in doc_keys:
                    errors.append(f"Positional index term '{term}' references unknown document '{doc}'.")
                doc_len = self.doc_lengths.get(doc, 0)
                for pos in positions:
                    if pos < 0 or pos >= doc_len:
                        errors.append(f"Invalid position {pos} for doc '{doc}' of length {doc_len}.")

        # Check statistics consistency
        if len(self.doc_lengths) != len(self.documents):
            errors.append("Mismatch between doc_lengths count and documents count.")

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "error_count": len(errors),
            "errors": errors,
            "index_version": self.index_version,
            "documents_checked": len(self.documents),
            "vocabulary_size": len(self.inverted_index)
        }

    # --- Index Serialization (Stage 11 Optimization) ---
    def save_index(self, path: Path = config.INDEX_CACHE_PATH) -> bool:
        """Safely serialize precomputed index to JSON for instant cold startup."""
        try:
            # Convert sets to sorted lists for JSON serialization
            serialized_inverted = {k: sorted(list(v)) for k, v in self.inverted_index.items()}
            data = {
                "version": self.index_version,
                "doc_lengths": self.doc_lengths,
                "term_counts": self.term_counts,
                "inverted_index": serialized_inverted,
                "positional_index": self.positional_index,
                "idf_cache": self.idf_cache,
                "index_stats": self.index_stats
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            print(f"[SearchEngine Warning] Failed to save serialized index: {e}")
            return False

    def load_index(self, path: Path = config.INDEX_CACHE_PATH) -> bool:
        """Load precomputed index from JSON, bypassing document processing."""
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.index_version = data.get("version", 1)
            self.doc_lengths = data.get("doc_lengths", {})
            self.term_counts = data.get("term_counts", {})
            self.positional_index = data.get("positional_index", {})
            self.idf_cache = data.get("idf_cache", {})
            self.index_stats = data.get("index_stats", {})
            
            # Convert lists back to sets
            self.inverted_index = {k: set(v) for k, v in data.get("inverted_index", {}).items()}
            self.clear_caches()
            return True
        except Exception as e:
            print(f"[SearchEngine Warning] Failed to load serialized index: {e}")
            return False

    def get_index_statistics(self) -> Dict[str, Any]:
        """Return the calculated index and collection statistics."""
        return self.index_stats

    # --- Precomputed TF and IDF Lookups (O(1)) ---
    def calculate_tf(self, term: str, filename: str) -> float:
        """O(1) precomputed Term Frequency lookup."""
        doc_len = self.doc_lengths.get(filename, 0)
        if doc_len == 0:
            return 0.0
        term_count = self.term_counts.get(filename, {}).get(term, 0)
        return term_count / doc_len

    def calculate_idf(self, term: str) -> float:
        """O(1) precomputed Inverse Document Frequency lookup from cache."""
        if config.IDF_CACHE_ENABLED and term in self.idf_cache:
            return self.idf_cache[term]
        
        df = self.doc_freq.get(term, len(self.inverted_index.get(term, set())))
        if df == 0 or len(self.documents) == 0:
            return 0.0
        val = math.log(len(self.documents) / df)
        self.idf_cache[term] = val
        return val

    # --- Health Check Endpoint Utility ---
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive search engine health and status check."""
        validation = self.validate_index()
        memory = get_memory_usage()
        return {
            "status": "healthy" if validation["is_valid"] else "degraded",
            "index_loaded": len(self.documents) > 0,
            "documents_indexed": len(self.documents),
            "vocabulary_size": len(self.inverted_index),
            "index_valid": validation["is_valid"],
            "index_version": self.index_version,
            "caching_enabled": config.CACHE_ENABLED,
            "query_cache_stats": self.query_cache.get_stats(),
            "fuzzy_cache_stats": self.fuzzy_cache.get_stats(),
            "memory_usage": memory
        }

    # --- Optimized Search Pipeline ---
    def search(
        self, 
        query: str, 
        log_analytics: bool = True, 
        top_k: Optional[int] = config.TOP_K_DEFAULT
    ) -> SearchResults:
        t_start = time.perf_counter()
        
        if not self.documents:
            res = SearchResults([], original_query=query)
            if log_analytics and config.ANALYTICS_ENABLED:
                record_search(query, 0, 0.0, query_type="empty")
            return res

        # 1. Check Query Cache (O(1) Bounded LRU Cache)
        cache_key = (query.strip(), self.index_version)
        if config.CACHE_ENABLED:
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                cached_copy = SearchResults(
                    list(cached_result),
                    original_query=cached_result.original_query,
                    did_you_mean=cached_result.did_you_mean,
                    corrections=cached_result.corrections,
                    timings=dict(cached_result.timings),
                    query_type=cached_result.query_type,
                    fuzzy_used=cached_result.fuzzy_used,
                    phrase_used=cached_result.phrase_used,
                    boolean_used=cached_result.boolean_used,
                    cache_hit=True,
                    candidate_reduction_pct=cached_result.candidate_reduction_pct
                )
                cached_copy.timings["total_search_duration"] = round(time.perf_counter() - t_start, 6)
                if log_analytics and config.ANALYTICS_ENABLED:
                    record_search(
                        query=query,
                        result_count=len(cached_copy),
                        search_duration=cached_copy.timings["total_search_duration"],
                        query_type=cached_copy.query_type,
                        fuzzy_used=cached_copy.fuzzy_used,
                        phrase_used=cached_copy.phrase_used,
                        boolean_used=cached_copy.boolean_used
                    )
                return cached_copy

        # 2. Tokenize and Parse the Boolean/Phrase Query
        t_parse_start = time.perf_counter()
        try:
            tokens = tokenize_query(query, process_text)
            if not tokens:
                res = SearchResults([], original_query=query)
                if log_analytics and config.ANALYTICS_ENABLED:
                    record_search(query, 0, 0.0, query_type="empty")
                return res
            parser = QueryParser(tokens)
            ast = parser.parse()
            if not ast:
                res = SearchResults([], original_query=query)
                if log_analytics and config.ANALYTICS_ENABLED:
                    record_search(query, 0, 0.0, query_type="empty")
                return res
        except ValueError as e:
            return {"error": str(e)}
        t_parse_end = time.perf_counter()
        query_parsing_time = t_parse_end - t_parse_start

        # Detect AST Features (Phrase & Boolean)
        has_phrase, has_boolean = detect_ast_features(ast)

        # 3. Fuzzy Term Resolution (with Bounded Cache & Length Filter)
        t_fuzzy_start = time.perf_counter()
        vocabulary = set(self.inverted_index.keys())
        resolved_ast, corrections = resolve_ast(ast, vocabulary, self.fuzzy_cache)
        t_fuzzy_end = time.perf_counter()
        term_resolution_time = t_fuzzy_end - t_fuzzy_start

        fuzzy_used = bool(corrections)
        did_you_mean = None
        if fuzzy_used:
            corrected_query = query
            for typo, corrected in corrections.items():
                pattern = re.compile(rf"\b{re.escape(typo)}\b", re.IGNORECASE)
                corrected_query = pattern.sub(corrected, corrected_query)
            did_you_mean = corrected_query

        # Determine Query Type label
        types = []
        if has_boolean:
            types.append("boolean")
        if has_phrase:
            types.append("phrase")
        if fuzzy_used:
            types.append("fuzzy")
        query_type = " + ".join(types) if types else "normal"

        # 4. Evaluate Expression (Candidate Filtering with Early Termination)
        t_retrieval_start = time.perf_counter()
        all_docs = set(self.documents.keys())
        matching_docs = evaluate_query(resolved_ast, self.inverted_index, all_docs, self.positional_index)
        t_retrieval_end = time.perf_counter()
        retrieval_time = t_retrieval_end - t_retrieval_start
        
        # Calculate Candidate Reduction Metric
        total_docs_count = len(self.documents)
        candidate_count = len(matching_docs)
        if total_docs_count > 0:
            candidate_reduction = round((1.0 - (candidate_count / total_docs_count)) * 100.0, 2)
        else:
            candidate_reduction = 0.0

        # 5. Extract Positive Terms for TF-IDF Ranking
        t_ranking_start = time.perf_counter()
        positive_terms = list(set(extract_positive_terms(resolved_ast)))
        
        # 6. Precomputed Scoring for Candidate Documents Only
        results = []
        for filename in matching_docs:
            score = 0.0
            for term in positive_terms:
                tf = self.calculate_tf(term, filename)
                idf = self.calculate_idf(term)
                score += (tf * idf)
                
            title = filename.replace(".txt", "").capitalize()
            raw_text = self.documents[filename]
            snippet = generate_snippet(raw_text, positive_terms)
            
            results.append({
                "filename": filename,
                "title": title,
                "score": score,
                "snippet": snippet
            })
            
        # 7. Sort / Top-K Optimization
        if top_k and len(results) > top_k:
            # Heap-based Top-K selection (O(n log k))
            ranked_results = heapq.nsmallest(
                top_k, 
                results, 
                key=lambda x: (-x["score"], x["filename"])
            )
        else:
            # Exact sorting
            ranked_results = sorted(results, key=lambda x: (-x["score"], x["filename"]))

        t_ranking_end = time.perf_counter()
        ranking_time = t_ranking_end - t_ranking_start

        total_search_duration = time.perf_counter() - t_start

        timings = {
            "query_parsing_time": round(query_parsing_time, 6),
            "term_resolution_time": round(term_resolution_time, 6),
            "retrieval_time": round(retrieval_time, 6),
            "ranking_time": round(ranking_time, 6),
            "total_search_duration": round(total_search_duration, 6)
        }

        search_result_obj = SearchResults(
            ranked_results, 
            original_query=query, 
            did_you_mean=did_you_mean, 
            corrections=corrections,
            timings=timings,
            query_type=query_type,
            fuzzy_used=fuzzy_used,
            phrase_used=has_phrase,
            boolean_used=has_boolean,
            cache_hit=False,
            candidate_reduction_pct=candidate_reduction
        )

        # Save to Bounded Query Cache
        if config.CACHE_ENABLED:
            self.query_cache.set(cache_key, search_result_obj)

        # 8. Record Analytics
        if log_analytics and config.ANALYTICS_ENABLED:
            record_search(
                query=query,
                result_count=len(ranked_results),
                search_duration=total_search_duration,
                query_parsing_time=query_parsing_time,
                term_resolution_time=term_resolution_time,
                retrieval_time=retrieval_time,
                ranking_time=ranking_time,
                query_type=query_type,
                fuzzy_used=fuzzy_used,
                phrase_used=has_phrase,
                boolean_used=has_boolean
            )
        
        return search_result_obj


# CLI Interface
def main():
    print("=" * 45)
    print("  MINI SEARCH ENGINE (CLI) Stage 11")
    print("  High-Performance Search & Index Optimization")
    print("=" * 45)

    engine = SearchEngine()
    stats = engine.get_index_statistics()
    print(f"Documents loaded:        {stats['total_documents']}")
    print(f"Unique vocabulary terms: {stats['vocabulary_size']}")
    print(f"Index build throughput:  {stats['throughput_docs_per_sec']} docs/sec")
    print(f"Query Cache & Precomp:   Active")

    while True:
        query = input("\nEnter search term (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            print("\nPlease enter a search term.")
            continue

        print("\nSearching...")
        results = engine.search(query)
        
        if isinstance(results, dict) and "error" in results:
            print(f"Error: {results['error']}")
            continue

        if getattr(results, "did_you_mean", None):
            print(f"\n* Did you mean: {results.did_you_mean}? *")
            print(f"Showing results for: {results.did_you_mean} (Original: {query})")
            
        cache_status = " [CACHE HIT]" if results.cache_hit else " [CACHE MISS]"
        print(f"\nResults found: {len(results)} (Latency: {results.timings.get('total_search_duration', 0)*1000:.3f} ms){cache_status}")
        print(f"Candidate Reduction: {results.candidate_reduction_pct}%")
        print(f"Timing Breakdown:")
        print(f"  - Parsing:        {results.timings.get('query_parsing_time', 0)*1000:.3f} ms")
        print(f"  - Term Lookup:    {results.timings.get('term_resolution_time', 0)*1000:.3f} ms")
        print(f"  - Retrieval:      {results.timings.get('retrieval_time', 0)*1000:.3f} ms")
        print(f"  - TF-IDF Ranking: {results.timings.get('ranking_time', 0)*1000:.3f} ms")
        print("-" * 40)

        for i, res in enumerate(results, start=1):
            print(f"  {i}. {res['filename']}")
            print(f"     TF-IDF Score: {res['score']:.4f}")

if __name__ == "__main__":
    main()
