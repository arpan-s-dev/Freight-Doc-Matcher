"""Thin SQL analytics layer over the matcher's real matched-load output.

The matcher already produces structured loads (broker, lane, rate, weight, match
quality). This package loads that output into a single DuckDB ``loads`` table,
exposes analytical SQL views (lane profitability, broker scorecard, exception
queue), and exports them to CSV/Parquet for Tableau / Power BI.

No invented data: every row is derived from a real ``Match`` or unmatched
``ExtractedDocument``. duckdb/pandas are imported lazily so the core matcher
install does not require the ``[bi]`` extra.
"""
