"""
Mini Search Engine - Stage 5
A command-line search engine using an Inverted Index, Text Processing, and TF-IDF Ranking.
"""

from pathlib import Path
import string
import math
from collections import Counter

# Predefined stop words for learning purposes
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "of",
    "in", "on", "to", "for", "with", "as", "at", "by", "from", "this",
    "that", "it"
}

def load_documents():
    """Load all .txt files from the documents/ directory."""
    documents_dir = Path(__file__).parent / "documents"
    if not documents_dir.exists() or not documents_dir.is_dir():
        print("Error: 'documents/' directory not found.")
        return {}

    txt_files = sorted(documents_dir.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in the 'documents/' directory.")
        return {}

    documents = {}
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                documents[file_path.name] = file.read()
        except Exception as e:
            print(f"Warning: Cannot read '{file_path.name}': {e}. Skipping.")

    return documents

def normalize_text(text):
    """Step 1 & 2: Lowercase and remove punctuation."""
    text = text.lower()
    for punc in string.punctuation:
        text = text.replace(punc, " ")
    return text

def tokenize_text(text):
    """Step 3: Split the processed text into individual words."""
    return text.split()

def remove_stop_words(tokens):
    """Step 4: Remove common, non-meaningful words."""
    filtered_tokens = []
    for token in tokens:
        if token not in STOP_WORDS:
            filtered_tokens.append(token)
    return filtered_tokens

def process_text(text, debug=False):
    """Complete Text Processing Pipeline."""
    normalized = normalize_text(text)
    tokens = tokenize_text(normalized)
    final_tokens = remove_stop_words(tokens)
    
    if debug:
        print(f"\n[DEBUG] Original: '{text}'")
        print(f"[DEBUG] Without punctuation: '{normalized}'")
        print(f"[DEBUG] Tokens: {tokens}")
        print(f"[DEBUG] After stop words: {final_tokens}\n")
        
    return final_tokens

def process_all_documents(documents):
    """Process all documents and store their tokens for ranking later."""
    processed_docs = {}
    for filename, content in documents.items():
        processed_docs[filename] = process_text(content)
    return processed_docs

def build_inverted_index(processed_documents):
    """Build an inverted index using processed tokens."""
    index = {}
    for filename, tokens in processed_documents.items():
        for token in tokens:
            if token not in index:
                index[token] = set()
            index[token].add(filename)
    return index

def calculate_tf(term, document_tokens):
    """
    Calculate Term Frequency (TF).
    TF = (Number of times term appears in document) / (Total terms in document)
    """
    total_terms = len(document_tokens)
    if total_terms == 0:
        return 0.0
    
    # Count occurrences of the specific term
    term_count = document_tokens.count(term)
    return term_count / total_terms

def calculate_document_frequency(term, inverted_index):
    """
    Calculate Document Frequency (DF).
    DF = Number of documents containing the term.
    """
    # Using the inverted index gives us the set of matching documents.
    # The length of this set is exactly the document frequency.
    if term in inverted_index:
        return len(inverted_index[term])
    return 0

def calculate_idf(term, inverted_index, total_documents):
    """
    Calculate Inverse Document Frequency (IDF).
    IDF = log(Total documents / Document frequency)
    """
    df = calculate_document_frequency(term, inverted_index)
    if df == 0:
        return 0.0
    
    return math.log(total_documents / df)

def calculate_tfidf(term, document_tokens, inverted_index, total_documents):
    """
    Calculate TF-IDF score for a specific term in a specific document.
    TF-IDF = TF * IDF
    """
    tf = calculate_tf(term, document_tokens)
    idf = calculate_idf(term, inverted_index, total_documents)
    return tf * idf

def score_document(document_tokens, query_terms, inverted_index, total_documents, debug=False):
    """
    Calculate the total TF-IDF score for a document against all query terms.
    """
    score = 0.0
    for term in query_terms:
        term_tfidf = calculate_tfidf(term, document_tokens, inverted_index, total_documents)
        score += term_tfidf
        
        if debug and term_tfidf > 0:
            tf = calculate_tf(term, document_tokens)
            idf = calculate_idf(term, inverted_index, total_documents)
            print(f"       [DEBUG] '{term}': TF = {tf:.4f}, IDF = {idf:.4f}, TF-IDF = {term_tfidf:.4f}")
            
    return score

def search_index(index, processed_documents, query, debug=False):
    """
    Search the inverted index and rank the matching documents using TF-IDF.
    """
    total_documents = len(processed_documents)
    
    # 1. Process the query using the exact same pipeline
    query_tokens = process_text(query, debug=debug)
    if not query_tokens:
        return []
        
    # Convert query terms into unique terms to avoid repeated query words artificially inflating scores.
    unique_query_terms = list(set(query_tokens))
    
    # 2. Find all matching documents (union of sets)
    matching_docs = set()
    for token in unique_query_terms:
        docs_for_token = index.get(token, set())
        matching_docs = matching_docs.union(docs_for_token)
    
    # 3. Calculate score for each matching document
    results = []
    for filename in matching_docs:
        if debug:
            print(f"  [DEBUG] Scoring {filename}:")
        doc_tokens = processed_documents[filename]
        score = score_document(doc_tokens, unique_query_terms, index, total_documents, debug=debug)
        results.append((filename, score))
    
    # 4. Sort results
    # Sort by score descending (highest first) and filename ascending (alphabetical tie-breaker)
    ranked_results = sorted(results, key=lambda x: (-x[1], x[0]))
    
    return ranked_results

def display_results(results):
    """Display ranked search results."""
    print(f"\nResults found: {len(results)}")
    if not results:
        print("\nNo documents found.")
        return

    print()
    for number, (filename, score) in enumerate(results, start=1):
        print(f"  {number}. {filename}")
        print(f"     TF-IDF Score: {score:.4f}")


def main():
    """Main function that runs Stage 5 of the Mini Search Engine."""
    print("=" * 30)
    print("  MINI SEARCH ENGINE (Stage 5)")
    print("=" * 30)

    documents = load_documents()
    if not documents:
        return
        
    print(f"\nDocuments loaded: {len(documents)}")

    # Pre-process documents and build index
    processed_documents = process_all_documents(documents)
    index = build_inverted_index(processed_documents)
    print(f"Unique terms indexed: {len(index)}")

    debug_mode = False
    query = input("\nEnter search term (or type '--debug' to turn on explanations): ").strip()

    if query.lower() == "--debug":
        debug_mode = True
        print("\n[Debug Mode Enabled]")
        query = input("Enter search term: ").strip()

    if not query:
        print("\nPlease enter a search term.")
        return

    print("\nSearching...")
    results = search_index(index, processed_documents, query, debug=debug_mode)
    display_results(results)

if __name__ == "__main__":
    main()
