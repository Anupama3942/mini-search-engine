from flask import Flask, render_template, request
import time
from search import SearchEngine

app = Flask(__name__)

# Initialize search engine in memory when the app starts
# This loads and processes documents only once!
search_engine = SearchEngine()

@app.route("/")
def index():
    """Homepage route showing search statistics."""
    doc_count = len(search_engine.documents)
    term_count = len(search_engine.inverted_index)
    return render_template("index.html", doc_count=doc_count, term_count=term_count)


@app.route("/search")
def search():
    """Search route handling queries and displaying ranked results."""
    query = request.args.get("q", "").strip()
    
    if not query:
        return render_template("results.html", error="Please enter a search term.", query="")

    # Time the search operation
    start_time = time.perf_counter()
    results = search_engine.search(query)
    end_time = time.perf_counter()
    
    search_time = round(end_time - start_time, 4)
    
    # Handle parsing errors from Stage 7
    if isinstance(results, dict) and "error" in results:
        return render_template("results.html", error=results["error"], query=query)
    
    return render_template(
        "results.html", 
        query=query, 
        results=results, 
        search_time=search_time
    )

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
