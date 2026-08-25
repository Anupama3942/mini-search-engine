"""
Mini Search Engine - Stage 10
Performance Benchmark Runner
"""

import time
from search import SearchEngine
from performance import calculate_percentiles, get_memory_usage

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


def run_benchmark(iterations_per_query: int = 10):
    print("=" * 65)
    print("      MINI SEARCH ENGINE - PERFORMANCE BENCHMARK SUITE")
    print("=" * 65)

    # 1. Index Build Benchmark
    t0 = time.perf_counter()
    engine = SearchEngine()
    build_time = time.perf_counter() - t0

    stats = engine.get_index_statistics()
    mem = get_memory_usage()

    print("\n--- [1] INDEX CONSTRUCTION METRICS ---")
    print(f"Total Documents Indexed:   {stats['total_documents']}")
    print(f"Vocabulary Size:           {stats['vocabulary_size']} terms")
    print(f"Total Tokens:              {stats['total_tokens']}")
    print(f"Total Postings:            {stats['total_postings']}")
    print(f"Avg Postings / Term:       {stats['avg_postings_per_term']}")
    print(f"Index Build Time:          {stats['build_time_seconds'] * 1000:.3f} ms")
    print(f"Indexing Throughput:       {stats['throughput_docs_per_sec']:.1f} docs/sec")

    print("\n--- [2] MEMORY CONSUMPTION (tracemalloc) ---")
    print(f"Current Heap Allocation:   {mem['current_kb']} KB ({mem['current_mb']} MB)")
    print(f"Peak Heap Allocation:      {mem['peak_kb']} KB ({mem['peak_mb']} MB)")

    # 2. Warm-up
    for category, queries in BENCHMARK_QUERIES.items():
        for q in queries:
            engine.search(q, log_analytics=False)

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

    print("\n--- [4] KEY BOTTLENECK OBSERVATIONS ---")
    fuzzy_avg = category_metrics.get("Fuzzy / Typo Queries", {}).get("avg", 0)
    normal_avg = category_metrics.get("Normal Queries", {}).get("avg", 0)
    if normal_avg > 0:
        ratio = round(fuzzy_avg / normal_avg, 2)
        print(f"* Fuzzy searches execute Levenshtein DP against the vocabulary, taking ~{ratio}x the time of normal lookups.")
    print(f"* Positional phrase checks scan positional postings for candidate documents.")
    print(f"* Exact lookups resolve in O(1) dictionary time, resulting in sub-millisecond responses.")
    print("=" * 65)

    return {
        "index_stats": stats,
        "memory": mem,
        "category_metrics": category_metrics,
        "overall_metrics": overall_pct
    }


if __name__ == "__main__":
    run_benchmark(iterations_per_query=15)
