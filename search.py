"""
Mini Search Engine - Stage 2
A simple command-line search engine using an Inverted Index.
"""

from pathlib import Path
import string


def load_documents():
    """
    Load all .txt files from the documents/ directory.
    (Same as Stage 1)
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


def tokenize_text(text):
    """
    Convert text into a list of normalized words.
    
    1. Convert to lowercase.
    2. Replace punctuation with spaces to avoid merging words.
    3. Split into words.
    """
    text = text.lower()
    
    # Replace punctuation with space
    for punc in string.punctuation:
        text = text.replace(punc, " ")
        
    # Split by whitespace
    words = text.split()
    return words


def build_inverted_index(documents):
    """
    Build an inverted index from the loaded documents.
    
    Instead of searching documents later, we pre-process them into a dictionary
    mapping each unique word to a set of filenames that contain it.
    
    Args:
        documents (dict): {filename: content}
        
    Returns:
        dict: {word: {filename1, filename2, ...}}
    """
    index = {}
    
    for filename, content in documents.items():
        words = tokenize_text(content)
        
        for word in words:
            if word not in index:
                index[word] = set()
            index[word].add(filename)
            
    return index


def search_index(index, query):
    """
    Search the inverted index for the given query.
    
    1. Tokenize the query to normalize it.
    2. Look it up in the index.
    3. Return the matching documents as a sorted list.
    """
    query_words = tokenize_text(query)
    
    if not query_words:
        return []
        
    # For now, we assume a single-word query for simplicity,
    # or just use the first token.
    query_word = query_words[0]
    
    # Dictionary lookup - O(1) average time!
    # If the word isn't in the index, return an empty set.
    matching_docs = index.get(query_word, set())
    
    # Return as a sorted list so results are consistent
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
    """Main function that runs Stage 2 of the Mini Search Engine."""
    print("=" * 30)
    print("  MINI SEARCH ENGINE")
    print("=" * 30)

    # 1. Load documents
    documents = load_documents()
    if not documents:
        return
        
    print(f"\n{len(documents)} documents loaded.")

    # 2. Build Inverted Index
    index = build_inverted_index(documents)
    print("Inverted index created.")
    print(f"Unique words indexed: {len(index)}")
    
    # (Optional Debugging View)
    # Uncomment to see a sample of the index:
    # print("\nIndex sample:")
    # for word in list(index.keys())[:5]:
    #     print(f"{word} -> {list(index[word])}")

    # 3. Ask user for search term
    query = input("\nEnter search term: ").strip()

    if not query:
        print("\nPlease enter a search term.")
        return

    # 4. Search using the Index
    print("\nSearching...")
    # Notice the difference from Stage 1: we pass the index, not the documents!
    results = search_index(index, query)

    # 5. Display results
    display_results(results)


if __name__ == "__main__":
    main()
