"""
Mini Search Engine - Stage 10 (Core Logic)
Now with Search Analytics, Performance Monitoring & Detailed Timing Instrumentation.
"""

from pathlib import Path
import string
import math
import html
import re
import time

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

# Predefined stop words
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "of",
    "in", "on", "to", "for", "with", "as", "at", "by", "from", "this",
    "that", "it"
}

def load_documents():
    documents_dir = Path(__file__).parent / "documents"
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

def normalize_text(text):
    text = text.lower()
    for punc in string.punctuation:
        text = text.replace(punc, " ")
    return text

def tokenize_text(text):
    return text.split()

def remove_stop_words(tokens):
    return [t for t in tokens if t not in STOP_WORDS]

def process_text(text):
    normalized = normalize_text(text)
    tokens = tokenize_text(normalized)
    return remove_stop_words(tokens)

def generate_snippet(document_text, query_terms):
    """
    Generate a short snippet and safely highlight query terms.
    """
    safe_text = html.escape(document_text)
    snippet = safe_text[:200]
    if len(safe_text) > 200:
        snippet += "..."
        
    for term in query_terms:
        safe_term = html.escape(term)
        pattern = re.compile(rf"\b{re.escape(safe_term)}\b", re.IGNORECASE)
        snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)

    return snippet

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
        boolean_used=False
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

def detect_ast_features(node):
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
    def __init__(self):
        self.documents = load_documents()
        self.processed_documents = {}
        self.inverted_index = {}
        self.positional_index = {}
        self.fuzzy_cache = {}
        self.index_stats = {}
        
        if self.documents:
            self._build_index()

    def _build_index(self):
        start_time = time.perf_counter()
        
        # Clear indexes and fuzzy cache upon rebuilding
        self.inverted_index.clear()
        self.positional_index.clear()
        self.fuzzy_cache.clear()
        self.processed_documents.clear()

        total_tokens = 0
        doc_lengths = {}

        # Process and store tokens
        for filename, content in self.documents.items():
            tokens = process_text(content)
            self.processed_documents[filename] = tokens
            doc_len = len(tokens)
            doc_lengths[filename] = doc_len
            total_tokens += doc_len
            
        # Build inverted index and positional index
        total_postings = 0
        total_stored_positions = 0

        for filename, tokens in self.processed_documents.items():
            for position, token in enumerate(tokens):
                # 1. Standard Inverted Index
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(filename)
                
                # 2. Positional Index
                # Structure: { term: { filename: [pos1, pos2] } }
                if token not in self.positional_index:
                    self.positional_index[token] = {}
                if filename not in self.positional_index[token]:
                    self.positional_index[token][filename] = []
                self.positional_index[token][filename].append(position)
                total_stored_positions += 1

        for term, postings in self.inverted_index.items():
            total_postings += len(postings)

        build_time = max(time.perf_counter() - start_time, 0.000001)
        doc_count = len(self.documents)
        vocab_size = len(self.inverted_index)

        largest_doc = max(doc_lengths.items(), key=lambda x: x[1]) if doc_lengths else ("None", 0)
        smallest_doc = min(doc_lengths.items(), key=lambda x: x[1]) if doc_lengths else ("None", 0)

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
            "throughput_docs_per_sec": round(doc_count / build_time, 2)
        }

    def get_index_statistics(self):
        """Return the calculated index and document collection statistics."""
        return self.index_stats

    def calculate_tf(self, term, document_tokens):
        total_terms = len(document_tokens)
        if total_terms == 0:
            return 0.0
        return document_tokens.count(term) / total_terms

    def calculate_idf(self, term):
        df = len(self.inverted_index.get(term, set()))
        if df == 0:
            return 0.0
        return math.log(len(self.documents) / df)

    def search(self, query: str, log_analytics: bool = True):
        t_start = time.perf_counter()
        
        if not self.documents:
            res = SearchResults([], original_query=query)
            if log_analytics:
                record_search(query, 0, 0.0, query_type="empty")
            return res
            
        # 1. Tokenize and Parse the Boolean/Phrase Query
        t_parse_start = time.perf_counter()
        try:
            tokens = tokenize_query(query, process_text)
            if not tokens:
                res = SearchResults([], original_query=query)
                if log_analytics:
                    record_search(query, 0, 0.0, query_type="empty")
                return res
            parser = QueryParser(tokens)
            ast = parser.parse()
            if not ast:
                res = SearchResults([], original_query=query)
                if log_analytics:
                    record_search(query, 0, 0.0, query_type="empty")
                return res
        except ValueError as e:
            return {"error": str(e)}
        t_parse_end = time.perf_counter()
        query_parsing_time = t_parse_end - t_parse_start

        # Detect AST Features (Phrase & Boolean)
        has_phrase, has_boolean = detect_ast_features(ast)

        # 2. Fuzzy Term Resolution (Typo Tolerance)
        t_fuzzy_start = time.perf_counter()
        vocabulary = set(self.inverted_index.keys())
        resolved_ast, corrections = resolve_ast(ast, vocabulary, self.fuzzy_cache)
        t_fuzzy_end = time.perf_counter()
        term_resolution_time = t_fuzzy_end - t_fuzzy_start

        fuzzy_used = bool(corrections)
        did_you_mean = None
        if fuzzy_used:
            # Reconstruct corrected query for display while preserving syntax
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

        # 3. Evaluate Expression (Retrieval & Filtering)
        t_retrieval_start = time.perf_counter()
        all_docs = set(self.documents.keys())
        matching_docs = evaluate_query(resolved_ast, self.inverted_index, all_docs, self.positional_index)
        t_retrieval_end = time.perf_counter()
        retrieval_time = t_retrieval_end - t_retrieval_start
        
        # 4. Extract Positive Terms for TF-IDF Ranking (using resolved terms)
        t_ranking_start = time.perf_counter()
        positive_terms = list(set(extract_positive_terms(resolved_ast)))
        
        # 5. Calculate Scores and Generate Results
        results = []
        for filename in matching_docs:
            doc_tokens = self.processed_documents[filename]
            
            score = 0.0
            for term in positive_terms:
                tf = self.calculate_tf(term, doc_tokens)
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
            
        # 6. Sort by score descending, then filename ascending (deterministic fallback)
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

        # 7. Record Analytics (Fault-Tolerant & Anonymous)
        if log_analytics:
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
        
        return SearchResults(
            ranked_results, 
            original_query=query, 
            did_you_mean=did_you_mean, 
            corrections=corrections,
            timings=timings,
            query_type=query_type,
            fuzzy_used=fuzzy_used,
            phrase_used=has_phrase,
            boolean_used=has_boolean
        )


# Keep CLI functionality
def main():
    print("=" * 40)
    print("  MINI SEARCH ENGINE (CLI) Stage 10")
    print("  Analytics & Performance Monitoring Active")
    print("=" * 40)

    engine = SearchEngine()
    stats = engine.get_index_statistics()
    print(f"Documents loaded:        {stats['total_documents']}")
    print(f"Unique vocabulary terms: {stats['vocabulary_size']}")
    print(f"Index build time:        {stats['build_time_seconds']:.4f}s ({stats['throughput_docs_per_sec']} docs/sec)")

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
            
        print(f"\nResults found: {len(results)} (Total search latency: {results.timings.get('total_search_duration', 0)*1000:.2f} ms)\n")
        print(f"Timing Breakdown:")
        print(f"  - Query Parsing:   {results.timings.get('query_parsing_time', 0)*1000:.3f} ms")
        print(f"  - Fuzzy Lookup:    {results.timings.get('term_resolution_time', 0)*1000:.3f} ms")
        print(f"  - Retrieval:       {results.timings.get('retrieval_time', 0)*1000:.3f} ms")
        print(f"  - TF-IDF Ranking:  {results.timings.get('ranking_time', 0)*1000:.3f} ms")
        print("-" * 35)

        for i, res in enumerate(results, start=1):
            print(f"  {i}. {res['filename']}")
            print(f"     TF-IDF Score: {res['score']:.4f}")

if __name__ == "__main__":
    main()
