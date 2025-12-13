"""Python access to the analytical SQL — returns pandas DataFrames.

KPIs are defined once in ``queries.py`` (SQL is the source of truth); these helpers
just run them so dashboards, the CLI, and tests share identical definitions.
"""

from matcher.analytics.load import DEFAULT_DB
from matcher.analytics.queries import QUERIES


def run_query(name: str, db_path: str = DEFAULT_DB):
    """Run a named query and return a pandas DataFrame."""
    import duckdb

    if name not in QUERIES:
        raise KeyError(f"unknown query '{name}'. Available: {', '.join(QUERIES)}")
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(QUERIES[name]).df()
    finally:
        con.close()


def query_on(con, name: str):
    """Run a named query against an existing connection (used in tests)."""
    return con.execute(QUERIES[name]).df()


def headline_kpis(db_path: str = DEFAULT_DB) -> dict:
    """Single-number KPIs for a dashboard scorecard row."""
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute("""
            SELECT
                COUNT(*)                                                  AS documents,
                SUM(CASE WHEN is_matched THEN 1 ELSE 0 END)               AS matched,
                ROUND(100.0 * AVG(CASE WHEN is_matched THEN 1 ELSE 0 END), 1) AS match_rate_pct,
                ROUND(SUM(CASE WHEN is_matched THEN rate ELSE 0 END), 2)  AS total_revenue,
                COUNT(DISTINCT lane)                                      AS lanes
            FROM loads
        """).fetchone()
    finally:
        con.close()
    return {"documents": row[0], "matched": row[1], "match_rate_pct": row[2],
            "total_revenue": row[3], "lanes": row[4]}
