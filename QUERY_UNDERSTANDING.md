# Query Understanding & Advanced Search (Stage 17)

## 1. Query Understanding Pipeline

The query understanding pipeline transforms raw, noisy user queries into rich, structured intent representations before retrieval and ranking:

```text
User Input: "  how to learn pythn web development?  "
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ 1. Query Validation & Normalization                    │
│    Unicode NFKC, noise punctuation stripped            │
│    -> "how to learn pythn web development"             │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ 2. Spell Correction with Confidence                    │
│    "pythn" -> "python" (Confidence: 0.95 >= 0.80)      │
│    -> "how to learn python web development"            │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ 3. Phrase & Field Extraction                           │
│    Quoted text detection, field filters (e.g. title:)  │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ 4. Important Term Extraction                           │
│    Positive concepts: ['learn', 'python', 'web', 'development']
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ 5. Synonym Expansion (Conservative Mode)               │
│    'web' -> ['website', 'internet'], 'learn' -> ['study']
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ 6. Intent Classification & Strategy Routing            │
│    Intent: "informational" -> Strategy: "semantic"     │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
          QueryRepresentation Object
```

---

## 2. Spell Correction & Confidence Formulation

To avoid over-correcting valid technical jargon, each candidate correction is assigned a confidence score $\in [0.0, 1.0]$:

$$\text{confidence} = \left(1 - \frac{\text{Levenshtein}(w_{\text{query}}, w_{\text{vocab}})}{\max(|w_{\text{query}}|, |w_{\text{vocab}}|)}\right) \times \left(0.8 + 0.2 \cdot \min\left(1.0, \frac{\ln(1 + \text{freq})}{\ln(6)}\right)\right)$$

- **Threshold**: Corrections are automatically applied only when $\text{confidence} \ge \text{SPELL\_CORRECTION\_THRESHOLD}$ (default: `0.80`).
- Otherwise, the query remains unchanged and a `"Did you mean: ..."` suggestion is provided.

---

## 3. Synonym Expansion Modes

Configured in `synonyms.json`:
- **`disabled`**: No synonym expansion.
- **`conservative` (Default)**: Expands only high-precision direct synonyms (e.g. `car` $\to$ `automobile`, `programming` $\to$ `coding`).
- **`aggressive`**: Expands broader domain-related terms (e.g. `python` $\to$ `django`, `flask`, `backend`).
- **Boolean Negation Safety**: Synonyms are **never** injected into `NOT` clauses or within exact quoted phrases (`"..."`).

---

## 4. Query Intent Classification & Adaptive Routing

| Intent Category | Query Characteristics | Trigger Rules | Recommended Strategy |
| :--- | :--- | :--- | :--- |
| **`boolean`** | Logical expressions | Contains `AND`, `OR`, `NOT`, `(` | `bm25` |
| **`phrase`** | Exact adjacency | Contains `"..."` | `bm25` |
| **`navigational`**| Document field specifiers | Contains `title:value` | `bm25` |
| **`informational`**| Natural language questions / guides | Contains `how`, `what`, `guide`, `tutorial`, `learn` | `semantic` |
| **`keyword`** | Short exact identifiers / code terms | Length $\le 2$ words (`python`, `sqlite3`) | `bm25` |
| **`mixed`** | Multi-concept keywords | General multi-term queries | `hybrid` ($\alpha=0.5$) |

---

## 5. Experimental Results (`experiment_query_understanding.py`)

Measured on the 25-query benchmark:

| Strategy / Mode | MAP | MRR | NDCG@5 | P@5 | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fixed BM25 (Baseline)** | **0.9884** | **1.0000** | **0.9900** | **0.4080** | 8.87 ms |
| **Query-Adaptive Routing (Auto)** | **0.8884** | **0.9000** | **0.8952** | **0.4080** | **6.56 ms** |
| **Query-Adaptive + Synonyms** | **0.8884** | **0.9000** | **0.8952** | **0.4080** | **6.76 ms** |
| **Fixed Hybrid ($\alpha=0.5$)** | **0.8816** | **0.9200** | **0.8806** | **0.3840** | 8.83 ms |
| **Fixed Semantic** | **0.7947** | **0.8267** | **0.8172** | **0.3840** | 11.03 ms |

---

## 6. Failure Analysis & Lessons Learned
1. **Synonym Over-Expansion**: Aggressive expansion on short keyword queries can dilute term specificity (e.g. expanding `"python"` to `"django"` may retrieve web documents for general Python queries). The `conservative` mode prevents this drift.
2. **Confidence Gating**: Without frequency-weighted confidence gating, rare terms could be falsely rewritten to common words. Gating at $\ge 0.80$ eliminates false corrections.
