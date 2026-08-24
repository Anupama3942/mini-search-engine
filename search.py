"""
Mini Search Engine - Stage 4
A command-line search engine using an Inverted Index, Text Processing, and basic Search Ranking.
"""

from pathlib import Path
import string
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

def calculate_score(document_tokens, query_terms, debug=False):
    """
    Calculate simple relevance score based on term frequency.
    
    1. Count term frequencies in the document using Counter.
    2. Add up the frequencies of the unique query terms.
    """
    # Create a frequency table of words in the document
    # Example: Counter(["python", "python", "programming"]) -> {"python": 2, "programming": 1}
    doc_freq = Counter(document_tokens)
    
    score = 0
    for term in query_terms:
        term_count = doc_freq.get(term, 0)
        score += term_count
        if debug and term_count > 0:
            print(f"       [DEBUG] '{term}': {term_count}")
            
    return score

def search_index(index, processed_documents, query, debug=False):
    """
    Search the inverted index and rank the matching documents.
    """
    # 1. Process the query using the exact same pipeline
    query_tokens = process_text(query, debug=debug)
    if not query_tokens:
        return []
        
    # Convert query terms into unique terms to avoid repeated query words artificially inflating scores.
    # Example: "python python programming" -> {"python", "programming"}
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
        score = calculate_score(doc_tokens, unique_query_terms, debug=debug)
        results.append((filename, score))
    
    # 4. Sort results
    # We sort by:
    #  - score descending (highest first) -> we use -x[1]
    #  - filename ascending (alphabetical tie-breaker) -> we use x[0]
    # sorted() uses the 'key' function to determine the sort order.
    # We return tuples of (-score, filename), and Python sorts them item by item.
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
        print(f"     Relevance Score: {score}")


def main():
    """Main function that runs Stage 4 of the Mini Search Engine."""
    print("=" * 30)
    print("  MINI SEARCH ENGINE (Stage 4)")
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
