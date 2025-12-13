"""Named analytical SQL over the ``loads`` table (DuckDB dialect).

These are the metrics a Fleet Ops / freight BI analyst would build dashboards on:
lane profitability, broker performance, match-quality mix, and the exception queue.
All read from the single ``loads`` table written by ``analytics.load``.
"""

QUERIES: dict[str, str] = {
    # Lane profitability — revenue and volume per origin->destination lane.
    "lane_summary": """
        SELECT lane,
               COUNT(*)                              AS loads,
               ROUND(SUM(rate), 2)                   AS total_revenue,
               ROUND(AVG(rate), 2)                   AS avg_rate,
               ROUND(AVG(weight_lbs), 0)             AS avg_weight_lbs,
               ROUND(AVG(rate) / NULLIF(AVG(weight_lbs), 0) * 1000, 2) AS rate_per_1k_lbs
        FROM loads
        WHERE is_matched AND lane IS NOT NULL
        GROUP BY lane
        ORDER BY total_revenue DESC NULLS LAST
    """,
    # Broker scorecard — volume, revenue, and match quality per broker.
    "broker_scorecard": """
        SELECT COALESCE(broker, '(unknown)')         AS broker,
               COUNT(*)                              AS documents,
               SUM(CASE WHEN is_matched THEN 1 ELSE 0 END) AS matched,
               ROUND(100.0 * AVG(CASE WHEN is_matched THEN 1 ELSE 0 END), 1) AS match_rate_pct,
               ROUND(AVG(CASE WHEN is_matched THEN match_score END), 1)      AS avg_match_score,
               ROUND(SUM(CASE WHEN is_matched THEN rate ELSE 0 END), 2)      AS revenue
        FROM loads
        GROUP BY 1
        ORDER BY revenue DESC NULLS LAST
    """,
    # Match-quality distribution — how confident the matcher was.
    "match_quality": """
        SELECT match_type,
               COUNT(*)                              AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
               ROUND(AVG(match_score), 1)            AS avg_score
        FROM loads
        GROUP BY match_type
        ORDER BY n DESC
    """,
    # Exception queue — unmatched docs and low-confidence matches needing review.
    "exception_queue": """
        SELECT load_number, broker, doc_type, lane, rate, match_type,
               ROUND(confidence, 2) AS confidence
        FROM loads
        WHERE NOT is_matched OR match_type = 'manual_review'
        ORDER BY confidence ASC
    """,
}
