"""
Mini Search Engine - Stage 10
Search Analytics Engine using SQLite
"""

import sqlite3
import datetime
import contextlib
from pathlib import Path
from typing import Dict, List, Any
from performance import calculate_percentiles

DEFAULT_DB_PATH = Path(__file__).parent / "analytics.db"


@contextlib.contextmanager
def get_db_connection(db_path: Path = DEFAULT_DB_PATH):
    """Context manager for SQLite connections ensuring clean closure."""
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initialize the search events table if it does not already exist."""
    try:
        with get_db_connection(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    result_count INTEGER NOT NULL,
                    search_duration REAL NOT NULL,
                    query_parsing_time REAL DEFAULT 0.0,
                    term_resolution_time REAL DEFAULT 0.0,
                    retrieval_time REAL DEFAULT 0.0,
                    ranking_time REAL DEFAULT 0.0,
                    query_type TEXT NOT NULL,
                    fuzzy_used INTEGER NOT NULL,
                    phrase_used INTEGER NOT NULL,
                    boolean_used INTEGER NOT NULL,
                    zero_result INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_events (normalized_query)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_zero ON search_events (zero_result)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON search_events (timestamp)")
            conn.commit()
    except Exception as e:
        print(f"[Analytics Warning] Could not initialize database: {e}")


# Initialize DB on module import
init_db()


def record_search(
    query: str,
    result_count: int,
    search_duration: float,
    query_parsing_time: float = 0.0,
    term_resolution_time: float = 0.0,
    retrieval_time: float = 0.0,
    ranking_time: float = 0.0,
    query_type: str = "normal",
    fuzzy_used: bool = False,
    phrase_used: bool = False,
    boolean_used: bool = False,
    db_path: Path = DEFAULT_DB_PATH
) -> bool:
    """
    Safely record a search event to SQLite using parameterized SQL.
    Privacy Guarantee: No IP addresses, user IDs, or personal info is collected.
    Fault Tolerance: Catches all exceptions so search operations are never interrupted.
    """
    try:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        normalized_query = query.strip().lower()
        zero_result = 1 if result_count == 0 else 0

        with get_db_connection(db_path) as conn:
            conn.execute("""
                INSERT INTO search_events (
                    timestamp, query, normalized_query, result_count,
                    search_duration, query_parsing_time, term_resolution_time,
                    retrieval_time, ranking_time, query_type,
                    fuzzy_used, phrase_used, boolean_used, zero_result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                query,
                normalized_query,
                result_count,
                search_duration,
                query_parsing_time,
                term_resolution_time,
                retrieval_time,
                ranking_time,
                query_type,
                1 if fuzzy_used else 0,
                1 if phrase_used else 0,
                1 if boolean_used else 0,
                zero_result
            ))
            conn.commit()
        return True
    except Exception as e:
        print(f"[Analytics Warning] Failed to record search event: {e}")
        return False


def get_summary_metrics(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Compute comprehensive aggregated search analytics:
      - Total search counts & total processing time
      - Latency statistics (Avg, Median/P50, P95, P99, Min, Max)
      - Zero-result rates
      - Feature usage rates (Fuzzy, Phrase, Boolean, Normal)
    """
    try:
        with get_db_connection(db_path) as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total_searches,
                    SUM(CASE WHEN zero_result = 1 THEN 1 ELSE 0 END) as zero_result_count,
                    SUM(CASE WHEN fuzzy_used = 1 THEN 1 ELSE 0 END) as fuzzy_count,
                    SUM(CASE WHEN phrase_used = 1 THEN 1 ELSE 0 END) as phrase_count,
                    SUM(CASE WHEN boolean_used = 1 THEN 1 ELSE 0 END) as boolean_count,
                    SUM(CASE WHEN boolean_used = 0 AND phrase_used = 0 AND fuzzy_used = 0 THEN 1 ELSE 0 END) as normal_count,
                    AVG(search_duration) as avg_duration,
                    SUM(search_duration) as total_duration
                FROM search_events
            """).fetchone()

            total_searches = row["total_searches"] or 0
            if total_searches == 0:
                return {
                    "total_searches": 0,
                    "avg_latency_ms": 0.0,
                    "median_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "min_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "total_search_time_s": 0.0,
                    "zero_result_count": 0,
                    "zero_result_rate": 0.0,
                    "fuzzy_usage_count": 0,
                    "fuzzy_usage_rate": 0.0,
                    "phrase_usage_count": 0,
                    "phrase_usage_rate": 0.0,
                    "boolean_usage_count": 0,
                    "boolean_usage_rate": 0.0,
                    "normal_usage_count": 0,
                    "normal_usage_rate": 0.0
                }

            zero_count = row["zero_result_count"] or 0
            fuzzy_count = row["fuzzy_count"] or 0
            phrase_count = row["phrase_count"] or 0
            boolean_count = row["boolean_count"] or 0
            normal_count = row["normal_count"] or 0
            total_duration = row["total_duration"] or 0.0

            cursor = conn.execute("SELECT search_duration FROM search_events")
            durations_ms = [r[0] * 1000.0 for r in cursor.fetchall()]
            pct = calculate_percentiles(durations_ms)

            return {
                "total_searches": total_searches,
                "avg_latency_ms": pct["avg"],
                "median_latency_ms": pct["p50"],
                "p95_latency_ms": pct["p95"],
                "p99_latency_ms": pct["p99"],
                "min_latency_ms": pct["min"],
                "max_latency_ms": pct["max"],
                "total_search_time_s": round(total_duration, 4),
                "zero_result_count": zero_count,
                "zero_result_rate": round((zero_count / total_searches) * 100.0, 2),
                "fuzzy_usage_count": fuzzy_count,
                "fuzzy_usage_rate": round((fuzzy_count / total_searches) * 100.0, 2),
                "phrase_usage_count": phrase_count,
                "phrase_usage_rate": round((phrase_count / total_searches) * 100.0, 2),
                "boolean_usage_count": boolean_count,
                "boolean_usage_rate": round((boolean_count / total_searches) * 100.0, 2),
                "normal_usage_count": normal_count,
                "normal_usage_rate": round((normal_count / total_searches) * 100.0, 2)
            }
    except Exception as e:
        print(f"[Analytics Warning] Failed to compute summary metrics: {e}")
        return {
            "total_searches": 0,
            "avg_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "total_search_time_s": 0.0,
            "zero_result_count": 0,
            "zero_result_rate": 0.0,
            "fuzzy_usage_count": 0,
            "fuzzy_usage_rate": 0.0,
            "phrase_usage_count": 0,
            "phrase_usage_rate": 0.0,
            "boolean_usage_count": 0,
            "boolean_usage_rate": 0.0,
            "normal_usage_count": 0,
            "normal_usage_rate": 0.0
        }


def get_top_queries(limit: int = 10, db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve the most frequent search queries with occurrence counts and average result count."""
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    normalized_query as query, 
                    COUNT(*) as count,
                    AVG(result_count) as avg_results,
                    AVG(search_duration * 1000.0) as avg_latency_ms
                FROM search_events
                GROUP BY normalized_query
                ORDER BY count DESC, normalized_query ASC
                LIMIT ?
            """, (limit,))
            return [
                {
                    "query": row["query"],
                    "count": row["count"],
                    "avg_results": round(row["avg_results"], 1),
                    "avg_latency_ms": round(row["avg_latency_ms"], 2)
                }
                for row in cursor.fetchall()
            ]
    except Exception as e:
        print(f"[Analytics Warning] Failed to get top queries: {e}")
        return []


def get_top_zero_result_queries(limit: int = 10, db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve the queries that yielded zero results most frequently."""
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    normalized_query as query, 
                    COUNT(*) as count
                FROM search_events
                WHERE zero_result = 1
                GROUP BY normalized_query
                ORDER BY count DESC, normalized_query ASC
                LIMIT ?
            """, (limit,))
            return [
                {"query": row["query"], "count": row["count"]}
                for row in cursor.fetchall()
            ]
    except Exception as e:
        print(f"[Analytics Warning] Failed to get zero result queries: {e}")
        return []


def get_query_type_distribution(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Retrieve search count breakdown by query category."""
    summary = get_summary_metrics(db_path)
    total = summary["total_searches"]
    return {
        "total": total,
        "normal": {"count": summary["normal_usage_count"], "rate": summary["normal_usage_rate"]},
        "boolean": {"count": summary["boolean_usage_count"], "rate": summary["boolean_usage_rate"]},
        "phrase": {"count": summary["phrase_usage_count"], "rate": summary["phrase_usage_rate"]},
        "fuzzy": {"count": summary["fuzzy_usage_count"], "rate": summary["fuzzy_usage_rate"]}
    }


def get_recent_searches(limit: int = 15, db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve recent search events."""
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    timestamp, query, result_count, 
                    search_duration, query_type, 
                    fuzzy_used, phrase_used, boolean_used
                FROM search_events
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            return [
                {
                    "timestamp": row["timestamp"],
                    "query": row["query"],
                    "result_count": row["result_count"],
                    "latency_ms": round(row["search_duration"] * 1000.0, 2),
                    "query_type": row["query_type"],
                    "fuzzy_used": bool(row["fuzzy_used"]),
                    "phrase_used": bool(row["phrase_used"]),
                    "boolean_used": bool(row["boolean_used"])
                }
                for row in cursor.fetchall()
            ]
    except Exception as e:
        print(f"[Analytics Warning] Failed to get recent searches: {e}")
        return []
