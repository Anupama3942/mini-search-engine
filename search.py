"""
Mini Search Engine - Stage 3
A command-line search engine using an Inverted Index and Text Processing.
"""

from pathlib import Path
import string


# A small set of predefined stop words for learning purposes
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "of",
    "in", "on", "to", "for", "with", "as", "at", "by", "from", "this",
    "that", "it"
}


def load_documents():
    """
    Load all .txt files from the documents/ directory.
    """
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
    """
    Step 1 & 2: Lowercase and remove punctuation.
    """
    text = text.lower()
    
    # Replace punctuation with space to avoid merging words
    # e.g., "Python,programming" -> "python programming"
    for punc in string.punctuation:
        text = text.replace(punc, " ")
        
    return text


def tokenize_text(text):
    """
    Step 3: Split the processed text into individual words.
    """
    return text.split()


def remove_stop_words(tokens):
    """
    Step 4: Remove common, non-meaningful words.
    """
    # Keep only tokens that are NOT in the STOP_WORDS set
    filtered_tokens = []
    for token in tokens:
        if token not in STOP_WORDS:
            filtered_tokens.append(token)
            
    return filtered_tokens


def process_text(text, debug=False):
    """
    Complete Text Processing Pipeline.
    Takes raw text and returns a list of normalized tokens.
    """
    if debug:
        print(f"\n[DEBUG] Original: '{text}'")
        
    normalized = normalize_text(text)
    if debug:
        print(f"[DEBUG] Without punctuation: '{normalized}'")
        
    tokens = tokenize_text(normalized)
    if debug:
        print(f"[DEBUG] Tokens: {tokens}")
        
    final_tokens = remove_stop_words(tokens)
    if debug:
        print(f"[DEBUG] After stop words: {final_tokens}\n")
        
    return final_tokens


def build_inverted_index(documents):
    """
    Build an inverted index using processed tokens.
    """
    index = {}
    
    for filename, content in documents.items():
        # Process the raw text from the document
        tokens = process_text(content)
        
        for token in tokens:
            if token not in index:
                index[token] = set()
            index[token].add(filename)
            
    return index


def search_index(index, query, debug=False):
    """
    Search the inverted index for the given query using the same text processing pipeline.
    Handles multi-word queries by taking the union of matching document sets.
    """
    # Important: Process the query using the EXACT same pipeline as the documents
    query_tokens = process_text(query, debug=debug)
    
    if not query_tokens:
        return []
        
    matching_docs = set()
    
    # Look up each processed query token in the index
    for token in query_tokens:
        docs_for_token = index.get(token, set())
        # Combine (union) the documents for this token with the overall matching docs
        matching_docs = matching_docs.union(docs_for_token)
    
    # Return as a sorted list for consistent output
    return sorted(list(matching_docs))


def display_results(results):
    """Display search results in a numbered list."""
    print(f"\nResults found: {len(results)}")

    if not results:
        print("\nNo documents found.")
        return

    print()
    for number, filename in enumerate(results, start=1):
        print(f"  {number}. {filename}")


def main():
    """Main function that runs Stage 3 of the Mini Search Engine."""
    print("=" * 30)
    print("  MINI SEARCH ENGINE (Stage 3)")
    print("=" * 30)

    # 1. Load documents
    documents = load_documents()
    if not documents:
        return
        
    print(f"\n{len(documents)} documents loaded.")

    # 2. Build Inverted Index
    index = build_inverted_index(documents)
    print("Inverted index created with Text Processing.")
    print(f"Unique processed words indexed: {len(index)}")

    # Optional: Developer demonstration flag
    debug_mode = False
    
    # 3. Ask user for search term
    query = input("\nEnter search term (or type '--debug' to turn on processing demo): ").strip()

    # Handle turning on debug mode
    if query.lower() == "--debug":
        debug_mode = True
        print("\n[Debug Mode Enabled]")
        query = input("Enter search term: ").strip()

    if not query:
        print("\nPlease enter a search term.")
        return

    # 4. Search using the Index
    print("\nSearching...")
    results = search_index(index, query, debug=debug_mode)

    # 5. Display results
    display_results(results)


if __name__ == "__main__":
    main()
