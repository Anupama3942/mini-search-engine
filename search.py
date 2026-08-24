"""
Mini Search Engine - Stage 6 (Core Logic)
Refactored into a SearchEngine class to be used by both CLI and Flask Web Interface.
"""

from pathlib import Path
import string
import math
import html

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
    # 1. Safely escape the HTML first to prevent XSS
    safe_text = html.escape(document_text)
    
    # Simple snippet extraction: just take the first 150 chars or try to center around the first match
    # For Stage 6, we'll keep it simple and just take the first 200 chars.
    snippet = safe_text[:200]
    if len(safe_text) > 200:
        snippet += "..."
        
    # 2. Highlight matching terms
    # We do a case-insensitive replacement
    for term in query_terms:
        # We need to escape the term in case the user typed something weird
        safe_term = html.escape(term)
        # Note: A true robust highlighter would use regex with word boundaries, 
        # but for this stage simple replacement (case-insensitive) is fine.
        # We replace the exact term with a highlighted version.
        # We must be careful not to double-highlight. 
        # For simplicity, we just use a basic replace on lowercased terms but keeping original case is better.
        # Since this is a beginner project, we can do a naive replace:
        
        # A simple case-insensitive replace without regex:
        import re
        # Use regex to replace case-insensitively, wrapping in <mark> tag
        pattern = re.compile(re.escape(safe_term), re.IGNORECASE)
        snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)

    return snippet

class SearchEngine:
    def __init__(self):
        self.documents = load_documents()
        self.processed_documents = {}
        self.inverted_index = {}
        
        if self.documents:
            self._build_index()

    def _build_index(self):
        # Process all documents
        for filename, content in self.documents.items():
            self.processed_documents[filename] = process_text(content)
            
        # Build inverted index
        for filename, tokens in self.processed_documents.items():
            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(filename)

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
            return []
            
        query_tokens = process_text(query)
        if not query_tokens:
            return []
            
        unique_query_terms = list(set(query_tokens))
        
        matching_docs = set()
        for token in unique_query_terms:
            matching_docs = matching_docs.union(self.inverted_index.get(token, set()))
            
        results = []
        for filename in matching_docs:
            doc_tokens = self.processed_documents[filename]
            
            score = 0.0
            for term in unique_query_terms:
                tf = self.calculate_tf(term, doc_tokens)
                idf = self.calculate_idf(term)
                score += (tf * idf)
                
            # Create a user-friendly title
            title = filename.replace(".txt", "").capitalize()
            
            # Generate snippet
            raw_text = self.documents[filename]
            snippet = generate_snippet(raw_text, unique_query_terms)
            
            results.append({
                "filename": filename,
                "title": title,
                "score": score,
                "snippet": snippet
            })
            
        # Sort by score descending, then filename ascending
        ranked_results = sorted(results, key=lambda x: (-x["score"], x["filename"]))
        return ranked_results


# Keep CLI functionality
def main():
    print("=" * 30)
    print("  MINI SEARCH ENGINE (CLI)")
    print("=" * 30)

    engine = SearchEngine()
    print(f"Documents loaded: {len(engine.documents)}")
    print(f"Unique terms indexed: {len(engine.inverted_index)}")

    query = input("\nEnter search term: ").strip()
    if not query:
        print("\nPlease enter a search term.")
        return

    print("\nSearching...")
    results = engine.search(query)
    
    print(f"\nResults found: {len(results)}\n")
    for i, res in enumerate(results, start=1):
        print(f"  {i}. {res['filename']}")
        print(f"     TF-IDF Score: {res['score']:.4f}")

if __name__ == "__main__":
    main()
