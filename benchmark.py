"""
Mini Search Engine - Stage 11
High-Performance Benchmark & Scaling Suite
"""

import time
import random
from search import SearchEngine
from performance import calculate_percentiles, get_memory_usage
import config

BENCHMARK_QUERIES = {
    "Normal Queries": [
        "python",
        "programming",
        "database",
        "web",
        "language",
        "learning"
    ],
    "Boolean Queries": [
        "python AND programming",
        "python OR java",
        "python AND NOT java",
        "(python OR java) AND programming",
        "web AND (python OR database)"
    ],
    "Phrase Queries": [
        '"python programming"',
        '"machine learning"',
        '"programming language"',
        '"python programming" AND language'
    ],
    "Fuzzy / Typo Queries": [
        "pythn",
        "programing",
        "databse",
        "pythn AND programing",
        "javscript OR pythn"
    ]
}

SAMPLE_VOCABULARY = [
    "python", "programming", "language", "data", "science", "database", "sql", 
    "nosql", "web", "development", "frontend", "backend", "framework", "flask", 
    "django", "fastapi", "machine", "learning", "artificial", "intelligence", 
    "deep", "neural", "network", "algorithm", "structure", "system", "distributed", 
    "cloud", "computing", "security", "encryption", "protocol", "http", "api", 
    "rest", "graphql", "microservices", "docker", "kubernetes", "testing"
]


def generate_synthetic_documents(count: int) -> dict:
    """Generate reproducible synthetic text documents for scaling benchmarks."""
    random.seed(42)
    docs = {}
    for i in range(1, count + 1):
        doc_len = random.randint(30, 80)
        words = [random.choice(SAMPLE_VOCABULARY) for _ in range(doc_len)]
        docs[f"synthetic_doc_{i:05d}.txt"] = " ".join(words)
    return docs


def run_benchmark(iterations_per_query: int = 15):
    print("=" * 70)
    print("       MINI SEARCH ENGINE - STAGE 11 OPTIMIZATION BENCHMARK")
    print("=" * 70)

    # 1. Index Build Benchmark
    t0 = time.perf_counter()
    engine = SearchEngine()
    build_time = time.perf_counter() - t0

    stats = engine.get_index_statistics()
    mem = get_memory_usage()

    print("\n--- [1] INDEX PRECOMPUTATION & CONSTRUCTION ---")
    print(f"Total Documents:           {stats['total_documents']}")
    print(f"Vocabulary Size:           {stats['vocabulary_size']} terms")
    print(f"Total Tokens:              {stats['total_tokens']}")
    print(f"Total Postings:            {stats['total_postings']}")
    print(f"Index Build Time:          {stats['build_time_seconds'] * 1000:.3f} ms")
    print(f"Indexing Throughput:       {stats['throughput_docs_per_sec']:.1f} docs/sec")
    print(f"Precomputed Lookups:       TF, IDF, doc_lengths, term_counts (O(1))")

    print("\n--- [2] MEMORY CONSUMPTION (tracemalloc) ---")
    print(f"Current Heap Allocation:   {mem['current_kb']} KB ({mem['current_mb']} MB)")
    print(f"Peak Heap Allocation:      {mem['peak_kb']} KB ({mem['peak_mb']} MB)")

    # 2. Cold vs Warm Cache Query Latency
    print(f"\n--- [3] QUERY LATENCY BENCHMARK ({iterations_per_query} runs per query) ---")
    print(f"{'Category':<22} | {'Avg (ms)':<9} | {'Median (ms)':<11} | {'P95 (ms)':<9} | {'Min (ms)':<9} | {'Max (ms)':<9}")
    print("-" * 75)

    all_latencies = []
    category_metrics = {}

    for category, queries in BENCHMARK_QUERIES.items():
        category_latencies = []
        for q in queries:
            for _ in range(iterations_per_query):
                t_start = time.perf_counter()
                engine.search(q, log_analytics=False)
                duration_ms = (time.perf_counter() - t_start) * 1000.0
                category_latencies.append(duration_ms)
                all_latencies.append(duration_ms)

        pct = calculate_percentiles(category_latencies)
        category_metrics[category] = pct

        print(f"{category:<22} | {pct['avg']:<9.3f} | {pct['p50']:<11.3f} | {pct['p95']:<9.3f} | {pct['min']:<9.3f} | {pct['max']:<9.3f}")

    overall_pct = calculate_percentiles(all_latencies)
    print("-" * 75)
    print(f"{'OVERALL ALL QUERIES':<22} | {overall_pct['avg']:<9.3f} | {overall_pct['p50']:<11.3f} | {overall_pct['p95']:<9.3f} | {overall_pct['min']:<9.3f} | {overall_pct['max']:<9.3f}")

    # 3. Cache Hit-Rate Metrics
    q_stats = engine.query_cache.get_stats()
    f_stats = engine.fuzzy_cache.get_stats()
    print(f"\n--- [4] CACHE PERFORMANCE ---")
    print(f"Query Result Cache:        {q_stats['hit_rate_pct']}% hit rate ({q_stats['hits']} hits / {q_stats['total_requests']} requests)")
    print(f"Fuzzy Typo Cache:          {f_stats['hit_rate_pct']}% hit rate ({f_stats['hits']} hits / {f_stats['total_requests']} requests)")

    # 4. Scaling Benchmark on Synthetic Datasets
    print(f"\n--- [5] CORPUS SCALABILITY BENCHMARK (Synthetic Documents) ---")
    print(f"{'Documents':<10} | {'Build Time':<12} | {'Throughput':<14} | {'Avg Query':<11} | {'P95 Query':<11} | {'Memory (MB)':<11}")
    print("-" * 75)

    scaling_sizes = [100, 500, 1000, 5000, 10000]
    scaling_results = []

    for size in scaling_sizes:
        synthetic_corpus = generate_synthetic_documents(size)
        t_b_start = time.perf_counter()
        synth_engine = SearchEngine(documents=synthetic_corpus)
        t_b_time = time.perf_counter() - t_b_start
        synth_stats = synth_engine.get_index_statistics()
        
        # Test sample queries
        sample_queries = ["python AND data", '"machine learning"', "pythn", "database OR cloud"]
        latencies = []
        for sq in sample_queries:
            for _ in range(5):
                t_q_start = time.perf_counter()
                synth_engine.search(sq, log_analytics=False)
                latencies.append((time.perf_counter() - t_q_start) * 1000.0)
                
        pct_s = calculate_percentiles(latencies)
        m_s = get_memory_usage()
        
        row_str = f"{size:<10} | {t_b_time*1000:<9.2f} ms | {synth_stats['throughput_docs_per_sec']:<9.1f} d/s | {pct_s['avg']:<8.3f} ms | {pct_s['p95']:<8.3f} ms | {m_s['current_mb']:<9.3f} MB"
        print(row_str)
        scaling_results.append({
            "size": size,
            "build_time_ms": t_b_time * 1000,
            "throughput": synth_stats['throughput_docs_per_sec'],
            "avg_ms": pct_s['avg'],
            "p95_ms": pct_s['p95'],
            "mem_mb": m_s['current_mb']
        })

    print("=" * 70)
    return {
        "index_stats": stats,
        "memory": mem,
        "category_metrics": category_metrics,
        "overall_metrics": overall_pct,
        "scaling_results": scaling_results
    }


if __name__ == "__main__":
    run_benchmark(iterations_per_query=15)
