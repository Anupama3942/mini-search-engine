"""
Mini Search Engine - Stage 9 (Core Logic)
Now using Fuzzy Search & Typo Tolerance with Levenshtein Distance.
"""

from pathlib import Path
import string
import math
import html
import re

from query_parser import (
    tokenize_query, 
    QueryParser, 
    evaluate_query, 
    extract_positive_terms,
    resolve_ast
)

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
    """List subclass holding search result items along with query correction metadata."""
    def __init__(self, items=None, original_query="", did_you_mean=None, corrections=None):
        super().__init__(items or [])
        self.original_query = original_query
        self.did_you_mean = did_you_mean
        self.corrections = corrections or {}

class SearchEngine:
    def __init__(self):
        self.documents = load_documents()
        self.processed_documents = {}
        self.inverted_index = {}
        self.positional_index = {}
        self.fuzzy_cache = {}
        
        if self.documents:
            self._build_index()

    def _build_index(self):
        # Clear indexes and fuzzy cache upon rebuilding
        self.inverted_index.clear()
        self.positional_index.clear()
        self.fuzzy_cache.clear()

        # Process and store tokens
        for filename, content in self.documents.items():
            self.processed_documents[filename] = process_text(content)
            
        # Build inverted index and positional index
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

    def search(self, query):
        if not self.documents:
            return SearchResults([], original_query=query)
            
        # 1. Tokenize and Parse the Boolean/Phrase Query
        try:
            tokens = tokenize_query(query, process_text)
            if not tokens:
                return SearchResults([], original_query=query)
            parser = QueryParser(tokens)
            ast = parser.parse()
            if not ast:
                return SearchResults([], original_query=query)
        except ValueError as e:
            return {"error": str(e)}
        
        # 2. Fuzzy Term Resolution (Typo Tolerance)
        # Compare unknown query terms against the indexed vocabulary
        vocabulary = set(self.inverted_index.keys())
        resolved_ast, corrections = resolve_ast(ast, vocabulary, self.fuzzy_cache)
        
        did_you_mean = None
        if corrections:
            # Reconstruct corrected query for display while preserving syntax
            corrected_query = query
            for typo, corrected in corrections.items():
                pattern = re.compile(rf"\b{re.escape(typo)}\b", re.IGNORECASE)
                corrected_query = pattern.sub(corrected, corrected_query)
            did_you_mean = corrected_query

        # 3. Evaluate Expression (Filtering)
        all_docs = set(self.documents.keys())
        matching_docs = evaluate_query(resolved_ast, self.inverted_index, all_docs, self.positional_index)
        
        # 4. Extract Positive Terms for TF-IDF Ranking (using resolved terms)
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
        
        return SearchResults(
            ranked_results, 
            original_query=query, 
            did_you_mean=did_you_mean, 
            corrections=corrections
        )


# Keep CLI functionality
def main():
    print("=" * 30)
    print("  MINI SEARCH ENGINE (CLI) Stage 9")
    print("=" * 30)

    engine = SearchEngine()
    print(f"Documents loaded: {len(engine.documents)}")
    print(f"Unique terms indexed: {len(engine.inverted_index)}")

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
            
        print(f"\nResults found: {len(results)}\n")
        for i, res in enumerate(results, start=1):
            print(f"  {i}. {res['filename']}")
            print(f"     TF-IDF Score: {res['score']:.4f}")

if __name__ == "__main__":
    main()
