"""
Mini Search Engine - Stage 1
A simple command-line search engine that searches through text documents.
"""

from pathlib import Path


def load_documents():
    """
    Load all .txt files from the documents/ directory.

    Uses pathlib to find the documents folder, reads every .txt file,
    and stores each filename and its content in a dictionary.

    Returns:
        dict: A dictionary where keys are filenames (str) and values
              are file contents (str). Returns an empty dict if no
              documents are found or the directory doesn't exist.
    """
    # Build the path to the documents folder relative to this script
    documents_dir = Path(__file__).parent / "documents"

    # Check if the documents directory exists
    if not documents_dir.exists():
        print("Error: 'documents/' directory not found.")
        print("Please create a 'documents/' folder and add .txt files.")
        return {}

    if not documents_dir.is_dir():
        print("Error: 'documents' exists but is not a directory.")
        return {}

    # Find all .txt files using glob
    txt_files = sorted(documents_dir.glob("*.txt"))

    # Check if any .txt files were found
    if not txt_files:
        print("No .txt files found in the 'documents/' directory.")
        return {}

    # Read each file and store its content in a dictionary
    documents = {}

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                # Use only the filename (not the full path) as the key
                documents[file_path.name] = content
        except PermissionError:
            print(f"Warning: Cannot read '{file_path.name}' (permission denied). Skipping.")
        except UnicodeDecodeError:
            print(f"Warning: Cannot read '{file_path.name}' (encoding error). Skipping.")

    return documents


def search_documents(documents, query):
    """
    Search for a query string inside all loaded documents.

    Performs a case-insensitive search by converting both the query
    and each document's content to lowercase before checking.

    Args:
        documents (dict): Dictionary of {filename: content} pairs.
        query (str): The search term entered by the user.

    Returns:
        list: A list of filenames (str) that contain the query.
    """
    # Convert the query to lowercase once (avoid doing it in every loop)
    query_lower = query.lower()

    # Check each document for the query
    results = []

    for filename, content in documents.items():
        # Convert content to lowercase for case-insensitive comparison
        if query_lower in content.lower():
            results.append(filename)

    return results


def display_results(results):
    """
    Display search results in a numbered list.

    Args:
        results (list): List of matching filenames.
    """
    print(f"\nResults found: {len(results)}")

    if not results:
        print("\nNo documents found.")
        return

    print()
    # enumerate starts counting from 1 for user-friendly numbering
    for number, filename in enumerate(results, start=1):
        print(f"  {number}. {filename}")


def main():
    """
    Main function that runs the Mini Search Engine.

    Flow:
    1. Display the title banner.
    2. Load all documents from the documents/ folder.
    3. Show how many documents were loaded.
    4. Ask the user for a search term.
    5. Validate the input.
    6. Search the documents.
    7. Display the results.
    """
    # Display the title
    print("=" * 30)
    print("  MINI SEARCH ENGINE")
    print("=" * 30)

    # Load documents
    documents = load_documents()

    # If no documents were loaded, exit early
    if not documents:
        print("\nNo documents available to search. Exiting.")
        return

    # Show how many documents are loaded
    print(f"\n{len(documents)} documents loaded.")

    # Ask the user for a search term
    query = input("\nEnter search term: ").strip()

    # Validate: check for empty input
    if not query:
        print("\nPlease enter a search term.")
        return

    # Perform the search
    print("\nSearching...")
    results = search_documents(documents, query)

    # Display the results
    display_results(results)


# This ensures main() only runs when the script is executed directly,
# not when it is imported as a module by another script.
if __name__ == "__main__":
    main()
