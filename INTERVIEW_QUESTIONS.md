# Search Engine - Technical Interview Questions

## Information Retrieval Fundamentals

**1. What is an inverted index and why is it used?**
An inverted index maps terms to the documents that contain them (like a book index), allowing $O(1)$ lookup time for search terms, as opposed to $O(N)$ scanning of all documents.

**2. What is a posting list?**
A posting list is the list of document IDs (and often term frequencies and positions) associated with a specific term in the inverted index.

**3. What is tokenization?**
Tokenization is the process of breaking unstructured text into discrete units (tokens/words), handling punctuation, casing, and special characters.

**4. Why do we remove stop words?**
Stop words (like "the", "and") appear in almost every document. Removing them reduces the size of the inverted index and speeds up query processing without losing much semantic meaning in classical IR.

## Ranking Algorithms

**5. How does Term Frequency (TF) affect search ranking?**
Higher term frequency indicates that a document is more relevant to that specific term.

**6. What is the purpose of Inverse Document Frequency (IDF)?**
IDF penalizes words that appear in many documents across the corpus, ensuring that rare, highly specific keywords contribute more to the search score than common words.

**7. Write the basic TF-IDF formula.**
$TF \times \log(N / df)$ where $N$ is total documents and $df$ is the number of documents containing the term.

**8. Why is BM25 considered better than basic TF-IDF?**
BM25 introduces term frequency saturation (more occurrences of a term have diminishing returns) and document length normalization (prevents long documents from unfairly dominating short ones).

**9. What does the $k_1$ parameter do in BM25?**
$k_1$ controls the term frequency saturation curve. A higher $k_1$ delays saturation, while a lower $k_1$ causes scores to flatline quickly after a few term occurrences.

**10. What does the $b$ parameter do in BM25?**
$b$ controls document length normalization. $b=1$ fully penalizes long documents, while $b=0$ applies no length penalty.

## Machine Learning for Search (LTR)

**11. What is Learning-to-Rank (LTR)?**
LTR uses machine learning models to combine multiple ranking signals (features) to compute an optimal relevance score for query-document pairs, replacing static formulas.

**12. Name three features you might extract for an LTR model.**
BM25 score, TF-IDF score, and document length (or exact term match ratio).

**13. What is the difference between Pointwise, Pairwise, and Listwise LTR?**
Pointwise evaluates a single document's relevance independently. Pairwise compares two documents and predicts which is better. Listwise optimizes the entire ranked list directly.

**14. What is Pointwise Logistic Regression in the context of search?**
It treats search ranking as a classification problem, predicting the probability (0 to 1) that a document is relevant to a query based on extracted features.

**15. What is feature ablation?**
The process of removing one feature at a time from an ML model to determine how much that specific feature contributes to overall performance.

## Semantic Search

**16. What are embeddings?**
Embeddings are dense mathematical vectors representing the semantic meaning of text, allowing systems to understand concepts rather than just matching characters.

**17. How does Cosine Similarity work?**
It calculates the cosine of the angle between two multi-dimensional vectors. A score of 1 means they point in the exact same direction (highly similar semantics).

**18. Why use semantic search over lexical search?**
Semantic search overcomes the "vocabulary mismatch problem" by matching concepts (e.g., finding "shoes" when querying for "sneakers").

**19. What is hybrid fusion in search?**
Combining scores from sparse (BM25) and dense (semantic) retrieval methods to get the best of exact keyword matching and broad conceptual matching.

**20. What is Alpha in hybrid retrieval?**
A weighting parameter used to balance the contribution of the dense score and sparse score ($Alpha \times Dense + (1-Alpha) \times Sparse$).

## Search Quality

**21. Define Precision and Recall.**
Precision is the percentage of returned documents that are relevant. Recall is the percentage of all relevant documents that were successfully returned.

**22. What is Mean Reciprocal Rank (MRR)?**
A metric that averages the reciprocal of the rank of the *first* relevant result across multiple queries. Highly sensitive to the top position.

**23. What is Mean Average Precision (MAP)?**
The mean of the Average Precision scores across queries. It rewards systems that return all relevant documents as high up the list as possible.

**24. Why is NDCG (Normalized Discounted Cumulative Gain) useful?**
NDCG handles graded relevance (e.g., a scale of 0 to 3) and applies a logarithmic discount to relevant items found lower in the search results.

**25. What is a confusion matrix?**
A table outlining True Positives, False Positives, True Negatives, and False Negatives, used to calculate metrics like Precision and Recall.

## System Design & Production

**26. Explain the Two-Stage Retrieval architecture.**
Stage 1 quickly retrieves a wide set of candidate documents using a cheap algorithm (BM25). Stage 2 reranks those candidates using an expensive, highly accurate model (LTR).

**27. Why use rate limiting on a search API?**
To protect the backend infrastructure from abuse, DDoS attacks, or excessive traffic spikes, ensuring fair resource allocation among users.

**28. What is the difference between a Liveness and Readiness probe?**
Liveness indicates if the application is running (or needs a restart). Readiness indicates if the application is fully loaded and ready to accept traffic (e.g., indexes are fully loaded).

## A/B Testing

**29. Why use deterministic hashing for A/B testing?**
Hashing a user ID to assign variants ensures the user always gets the same experience across sessions without needing to maintain a persistent state database.

**30. What is a confidence interval in A/B testing?**
A statistical range of values that likely contains the true impact of an experiment, helping to understand the uncertainty of the results.
