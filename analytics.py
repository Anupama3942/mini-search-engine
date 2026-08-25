"""
Mini Search Engine - Stage 10, 16 & 20
Advanced Search Analytics & Experimentation Engine using SQLite
"""

import sqlite3
import datetime
import contextlib
from pathlib import Path
from typing import Dict, List, Any, Optional

import config
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
    """Initialize search and click event tables if they do not already exist."""
    try:
        with get_db_connection(db_path) as conn:
            # 1. Search Events Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    search_method TEXT DEFAULT 'bm25',
                    experiment_id TEXT,
                    variant TEXT,
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
            # Schema Migration for existing databases
            existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(search_events)").fetchall()}
            for col_name, col_type in [
                ("request_id", "TEXT"),
                ("session_id", "TEXT"),
                ("search_method", "TEXT DEFAULT 'bm25'"),
                ("experiment_id", "TEXT"),
                ("variant", "TEXT")
            ]:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE search_events ADD COLUMN {col_name} {col_type}")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_events (normalized_query)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_zero ON search_events (zero_result)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON search_events (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_req ON search_events (request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp ON search_events (experiment_id, variant)")

            # 2. Click Events Table (Stage 20 Funnel & CTR tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS click_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    experiment_id TEXT,
                    variant TEXT,
                    search_method TEXT DEFAULT 'bm25'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_click_req ON click_events (request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_click_time ON click_events (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_click_pos ON click_events (position)")

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
    search_method: str = "bm25",
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    variant: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH
) -> bool:
    """
    Safely record a search event to SQLite using parameterized SQL.
    Privacy Guarantee: Respects PRIVACY_MASK_QUERIES if enabled.
    """
    if not config.ANALYTICS_ENABLED:
        return True

    try:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        clean_q = query.strip().lower()
        if config.PRIVACY_MASK_QUERIES:
            clean_q = f"query_len_{len(clean_q)}"

        zero_result = 1 if result_count == 0 else 0

        with get_db_connection(db_path) as conn:
            conn.execute("""
                INSERT INTO search_events (
                    request_id, session_id, timestamp, query, normalized_query,
                    search_method, experiment_id, variant, result_count,
                    search_duration, query_parsing_time, term_resolution_time,
                    retrieval_time, ranking_time, query_type,
                    fuzzy_used, phrase_used, boolean_used, zero_result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id,
                session_id,
                timestamp,
                query if not config.PRIVACY_MASK_QUERIES else "[MASKED]",
                clean_q,
                search_method,
                experiment_id,
                variant,
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


def record_click(
    request_id: str,
    doc_id: str,
    position: int,
    session_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    variant: Optional[str] = None,
    search_method: str = "bm25",
    db_path: Path = DEFAULT_DB_PATH
) -> bool:
    """Record a user click event for search result attribution."""
    if not config.ANALYTICS_ENABLED:
        return True

    try:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with get_db_connection(db_path) as conn:
            conn.execute("""
                INSERT INTO click_events (
                    request_id, session_id, timestamp, doc_id,
                    position, experiment_id, variant, search_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id,
                session_id,
                timestamp,
                doc_id,
                position,
                experiment_id,
                variant,
                search_method
            ))
            conn.commit()
        return True
    except Exception as e:
        print(f"[Analytics Warning] Failed to record click event: {e}")
        return False


def get_summary_metrics(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Compute comprehensive aggregated search and performance analytics."""
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


def get_ctr_analytics(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Compute Click-Through Rate (CTR) metrics:
      - Overall CTR %
      - CTR by Rank Position (1..5)
      - CTR by Search Ranking Method
    """
    try:
        with get_db_connection(db_path) as conn:
            # 1. Total Searches & Clicks
            s_row = conn.execute("SELECT COUNT(*) as count FROM search_events").fetchone()
            c_row = conn.execute("SELECT COUNT(*) as count FROM click_events").fetchone()

            total_searches = s_row["count"] or 0
            total_clicks = c_row["count"] or 0
            overall_ctr = round((total_clicks / total_searches) * 100.0, 2) if total_searches > 0 else 0.0

            # 2. Clicks by Position
            pos_cursor = conn.execute("""
                SELECT position, COUNT(*) as clicks
                FROM click_events
                GROUP BY position
                ORDER BY position ASC
                LIMIT 10
            """)
            position_clicks = [{"position": r["position"], "clicks": r["clicks"]} for r in pos_cursor.fetchall()]

            # 3. CTR by Search Method
            method_cursor = conn.execute("""
                SELECT 
                    s.search_method,
                    COUNT(DISTINCT s.id) as searches,
                    COUNT(DISTINCT c.id) as clicks
                FROM search_events s
                LEFT JOIN click_events c ON s.request_id = c.request_id
                GROUP BY s.search_method
            """)
            method_ctr = []
            for r in method_cursor.fetchall():
                s_cnt = r["searches"]
                c_cnt = r["clicks"]
                ctr_val = round((c_cnt / s_cnt) * 100.0, 2) if s_cnt > 0 else 0.0
                method_ctr.append({
                    "search_method": r["search_method"] or "bm25",
                    "searches": s_cnt,
                    "clicks": c_cnt,
                    "ctr_pct": ctr_val
                })

            return {
                "total_searches": total_searches,
                "total_clicks": total_clicks,
                "overall_ctr_pct": overall_ctr,
                "clicks_by_position": position_clicks,
                "ctr_by_method": method_ctr
            }
    except Exception as e:
        print(f"[Analytics Warning] Failed to compute CTR analytics: {e}")
        return {
            "total_searches": 0,
            "total_clicks": 0,
            "overall_ctr_pct": 0.0,
            "clicks_by_position": [],
            "ctr_by_method": []
        }


def get_online_experiment_summary(experiment_id: str, db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Retrieve online telemetry comparison for a specific A/B experiment."""
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    variant,
                    COUNT(DISTINCT s.id) as searches,
                    COUNT(DISTINCT c.id) as clicks,
                    AVG(s.search_duration * 1000.0) as avg_latency_ms
                FROM search_events s
                LEFT JOIN click_events c ON s.request_id = c.request_id
                WHERE s.experiment_id = ?
                GROUP BY variant
            """, (experiment_id,))

            variants_data = {}
            for r in cursor.fetchall():
                v = r["variant"] or "unknown"
                searches = r["searches"]
                clicks = r["clicks"]
                ctr = round((clicks / searches) * 100.0, 2) if searches > 0 else 0.0
                variants_data[v] = {
                    "searches": searches,
                    "clicks": clicks,
                    "ctr_pct": ctr,
                    "avg_latency_ms": round(r["avg_latency_ms"] or 0.0, 2)
                }

            return {
                "experiment_id": experiment_id,
                "variants": variants_data,
                "status": "active" if variants_data else "no_traffic"
            }
    except Exception as e:
        print(f"[Analytics Warning] Failed to compute online experiment summary: {e}")
        return {"experiment_id": experiment_id, "variants": {}, "status": "error"}


def get_top_queries(limit: int = 10, db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve most frequent search queries with counts and latency."""
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
    """Retrieve queries that yielded zero results most frequently."""
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
                    fuzzy_used, phrase_used, boolean_used,
                    search_method, experiment_id, variant
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
                    "search_method": row["search_method"] or "bm25",
                    "experiment_id": row["experiment_id"],
                    "variant": row["variant"],
                    "fuzzy_used": bool(row["fuzzy_used"]),
                    "phrase_used": bool(row["phrase_used"]),
                    "boolean_used": bool(row["boolean_used"])
                }
                for row in cursor.fetchall()
            ]
    except Exception as e:
        print(f"[Analytics Warning] Failed to get recent searches: {e}")
        return []


def cleanup_old_analytics(retention_days: int = config.ANALYTICS_RETENTION_DAYS, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Delete search and click records older than retention threshold for privacy compliance."""
    try:
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=retention_days)).isoformat()
        with get_db_connection(db_path) as conn:
            c1 = conn.execute("DELETE FROM search_events WHERE timestamp < ?", (cutoff,)).rowcount
            c2 = conn.execute("DELETE FROM click_events WHERE timestamp < ?", (cutoff,)).rowcount
            conn.commit()
            return c1 + c2
    except Exception as e:
        print(f"[Analytics Warning] Failed to run retention cleanup: {e}")
        return 0
