"""
Mini Search Engine - Stage 11
Fuzzy Search & Typo Tolerance using Levenshtein Distance & Optimization
"""

def levenshtein_distance(a: str, b: str) -> int:
    """
    Calculate the Levenshtein distance between two strings using dynamic programming.
    Allowed operations:
      1. Insertion
      2. Deletion
      3. Substitution
    Case-insensitive comparison.
    """
    a = a.lower()
    b = b.lower()
    m, n = len(a), len(b)

    # Initialize (m + 1) x (n + 1) DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases:
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion from a
                    dp[i][j - 1],      # insertion into a
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n]


def levenshtein_distance_optimized(a: str, b: str) -> int:
    """
    Memory-optimized two-row implementation of Levenshtein distance.
    Space Complexity: O(min(m, n))
    """
    a = a.lower()
    b = b.lower()
    if len(a) < len(b):
        a, b = b, a

    m, n = len(a), len(b)
    previous_row = list(range(n + 1))
    current_row = [0] * (n + 1)

    for i in range(1, m + 1):
        current_row[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                current_row[j] = previous_row[j - 1]
            else:
                current_row[j] = 1 + min(
                    previous_row[j],      # deletion
                    current_row[j - 1],   # insertion
                    previous_row[j - 1]   # substitution
                )
        previous_row = list(current_row)

    return previous_row[n]


def max_edit_distance(term: str) -> int:
    """
    Determine the maximum allowed edit distance based on term length:
      - Length 1-3: distance 0 (no fuzzy matching for short words)
      - Length 4-6: distance <= 1
      - Length 7+:  distance <= 2
    """
    length = len(term)
    if length <= 3:
        return 0
    elif length <= 6:
        return 1
    else:
        return 2


def find_fuzzy_matches(term: str, vocabulary, max_distance: int = None):
    """
    Find matching candidate terms from the vocabulary within the edit distance threshold.
    Optimizations:
      1. Length difference pre-filter: skips candidates with |len(cand) - len(term)| > max_dist
      2. Deterministic sorting: by distance ascending, then term
    """
    normalized_term = term.lower()
    if max_distance is None:
        max_dist = max_edit_distance(normalized_term)
    else:
        max_dist = max_distance

    if max_dist == 0:
        return []

    candidates = []
    term_len = len(normalized_term)
    for candidate in vocabulary:
        cand_lower = candidate.lower()
        
        # Optimization: Length filter (skips impossible comparisons immediately)
        if abs(len(cand_lower) - term_len) > max_dist:
            continue

        dist = levenshtein_distance(normalized_term, cand_lower)
        if dist <= max_dist:
            candidates.append({
                "term": candidate,
                "distance": dist
            })

    candidates.sort(key=lambda x: (x["distance"], x["term"]))
    return candidates


def resolve_term(term: str, vocabulary, cache = None):
    """
    Resolve a single term:
      - 1. Exact match check (O(1))
      - 2. Bounded Cache lookup (O(1))
      - 3. Levenshtein Candidate Search (with length pre-filtering)
    """
    normalized = term.lower()
    
    # 1. Exact Match First (Optimization)
    if normalized in vocabulary:
        return normalized, False, 0

    # 2. Check Cache
    if cache is not None:
        if hasattr(cache, "get"):
            cached = cache.get(normalized)
        else:
            cached = cache.get(normalized) if isinstance(cache, dict) else None
        if cached is not None:
            return cached

    # 3. Fuzzy Lookup
    matches = find_fuzzy_matches(normalized, vocabulary)
    if matches:
        best_candidate = matches[0]["term"]
        dist = matches[0]["distance"]
        result = (best_candidate, True, dist)
    else:
        result = (normalized, False, 0)

    # 4. Save to Cache
    if cache is not None:
        if hasattr(cache, "set"):
            cache.set(normalized, result)
        elif isinstance(cache, dict):
            cache[normalized] = result

    return result
