# Search Experimentation Platform & A/B Testing Guide (Stage 20)

## 1. Experimentation Architecture Overview

The search engine includes a built-in search experimentation platform designed to evaluate ranking models, query understanding configurations, and two-stage retrieval pipelines:

```text
User Request / Query
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 1. Experimentation Registry                            │
│    Active Experiment Lookup (e.g. bm25_vs_hybrid)      │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ 2. Deterministic Variant Hashing                       │
│    SHA-256(experiment_id:session_id) % 100            │
│    Bucket < 50 -> Variant A (BM25)                     │
│    Bucket >= 50 -> Variant B (Hybrid)                  │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ 3. Pipeline Execution & Telemetry Attribution          │
│    Record Impression: search_events (exp_id, variant)  │
│    Record Click: click_events (pos, exp_id, variant)   │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ 4. Statistical Analysis & Evaluation                   │
│    Offline: MAP, MRR, NDCG@5, NDCG@10, P@5, Latency    │
│    Online: CTR by position, Zero-Result Rate           │
│    Metrics: Diff, Uplift %, 95% CI, Welch's t-test     │
└────────────────────────────────────────────────────────┘
```

---

## 2. Deterministic Variant Assignment

To guarantee consistent user experience across repeat queries and navigation, variant assignment is deterministic:

$$\text{bucket} = \text{SHA-256}(\text{experiment\_id} + \text{":"} + \text{session\_id}) \pmod{100}$$

- **Traffic Split (e.g. 50/50)**:
  - $\text{bucket} < 50 \implies \text{Variant A (Control)}$
  - $\text{bucket} \ge 50 \implies \text{Variant B (Treatment)}$
- **Traffic Gating (e.g. 10% rollouts)**:
  - $\text{bucket} \ge \text{traffic\_percentage} \implies \text{Excluded (Default Ranking)}$.

---

## 3. Statistical Analysis & Confidence Intervals

### Formulas:
- **Mean**: $\mu = \frac{1}{N} \sum x_i$
- **Variance**: $s^2 = \frac{1}{N-1} \sum (x_i - \mu)^2$
- **Mean Difference**: $\Delta = \mu_B - \mu_A$
- **Relative Uplift**: $\frac{\Delta}{\mu_A} \times 100\%$
- **Standard Error**: $SE = \sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}$
- **95% Confidence Interval**: $[\Delta - 1.96 \cdot SE, \Delta + 1.96 \cdot SE]$
- **Welch's t-statistic**: $t = \frac{\Delta}{SE}$ (Significant if $|t| \ge 1.96$ and $N \ge 10$).

---

## 4. Offline A/B Experiment Results (`python run_experiment.py`)

Measured across 25 ground-truth benchmark queries:

### Experiment 1: BM25 vs Hybrid Search Fusion (`bm25_vs_hybrid`)
- **Primary Metric**: `NDCG@5`
- **Variant A (BM25)**: NDCG@5 = `0.9900`, MAP = `0.9884`, MRR = `1.0000`, Latency = `8.29 ms`
- **Variant B (Hybrid)**: NDCG@5 = `0.8806`, MAP = `0.8816`, MRR = `0.9200`, Latency = `11.06 ms`
- **Difference**: $\Delta = -0.1094$ ($-11.05\%$ uplift, 95% CI: `[-0.2188, 0.0000]`, $p > 0.05$, not statistically significant).

### Experiment 2: BM25 vs Dense Semantic Search (`bm25_vs_semantic`)
- **Primary Metric**: `MAP`
- **Variant A (BM25)**: MAP = `0.9884`, NDCG@5 = `0.9900`
- **Variant B (Semantic)**: MAP = `0.8147`, NDCG@5 = `0.8320`
- **Difference**: $\Delta = -0.1737$ ($-17.57\%$ uplift, 95% CI: `[-0.2949, -0.0524]`, **Statistically Significant** (*)).
- **Conclusion**: BM25 is significantly superior for technical keyword queries.

### Experiment 3: Two-Stage BM25 $\to$ LTR vs Hybrid $\to$ LTR (`bm25_ltr_vs_hybrid_ltr`)
- **Primary Metric**: `NDCG@5`
- **Variant A (BM25 $\to$ LTR)**: NDCG@5 = `0.9915`, MAP = `0.9920`, Latency = `2.29 ms`
- **Variant B (Hybrid $\to$ LTR)**: NDCG@5 = `0.7717`, MAP = `0.7305`, Latency = `5.63 ms`
- **Difference**: $\Delta = -0.2198$ ($-22.17\%$ uplift, 95% CI: `[-0.3463, -0.0933]`, **Statistically Significant** (*)).

---

## 5. Privacy Guarantees & Retention Policy

- **No Personal Identifiers**: No IP addresses, device fingerprints, or personally identifiable information (PII) are stored.
- **Configurable Masking**: `PRIVACY_MASK_QUERIES=true` masks query text into generic length buckets.
- **Automated Retention Cleanup**: `cleanup_old_analytics(days=30)` deletes events older than `ANALYTICS_RETENTION_DAYS`.
