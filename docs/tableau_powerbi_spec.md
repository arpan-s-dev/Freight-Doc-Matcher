# Fleet / Freight Operations Dashboard — Tableau & Power BI Spec

This spec documents a BI dashboard built on the analytics exports. The matcher writes
Tableau/Power-BI-ready files via:

```bash
matcher analytics build samples/input --matcher heuristic
matcher analytics export --format parquet      # -> output/exports/*.parquet
# or: --format csv
```

Both tools connect directly to the Parquet/CSV files (or to the DuckDB database via
the DuckDB connector). The base `loads.parquet` plus the four view exports
(`lane_summary`, `broker_scorecard`, `match_quality`, `exception_queue`) are the data
sources.

---

## Data source

- **Primary:** `output/exports/loads.parquet` (grain: one matched load / document).
- **Pre-aggregated:** the four view Parquets, for cards and ranked tables.
- A Power BI `.pbids` connecting to the exports folder can be added for one-click open.

## Calculated fields

| name                | definition                                              |
|---------------------|---------------------------------------------------------|
| Match Rate %        | `SUM([is_matched]) / COUNT([load_number]) * 100`        |
| Rate per 1k lbs     | `AVG([rate]) / AVG([weight_lbs]) * 1000`                |
| Needs Review        | `[match_type] = "manual_review" OR NOT [is_matched]`    |
| Revenue             | `SUM(IF [is_matched] THEN [rate] END)`                  |

## Dashboard layout

**Row 1 — KPI scorecards** (from `headline_kpis` / `match_quality`):
- Documents processed · Match rate % · Total revenue · Distinct lanes · % needs review.

**Row 2 — Lane profitability**
- *Worksheet:* horizontal bar, **Lane** (rows) × **Revenue** (length), color = Rate
  per 1k lbs, sorted desc. Source: `lane_summary`.
- *Map (optional):* origin→destination paths sized by load volume (geocode the ZIPs).

**Row 3 — Broker scorecard**
- *Worksheet:* table with data bars — Broker × {Documents, Match Rate %, Avg Match
  Score, Revenue}. Conditional formatting: Match Rate % red < 80, green ≥ 95. Source:
  `broker_scorecard`.

**Row 4 — Match quality & exceptions**
- *Donut:* share of `exact_load` / `fuzzy` / `manual_review` / `unmatched`. Source:
  `match_quality`.
- *Exception queue table:* `exception_queue` view, ordered by confidence asc — the
  operational worklist of documents a human must reconcile.

## Filters (dashboard-wide)

- Broker (multi-select), Date range (pickup_date), Lane (search), Match type.

## Why these views

The dashboard answers the Fleet Ops questions the role cares about: *Which lanes and
brokers drive revenue? How reliable is each broker's paperwork? How big is the manual
reconciliation backlog, and which documents need attention first?* Every number traces
back to a real matched document — the SQL definitions live in
`src/matcher/analytics/queries.py` so the BI tool and the Python/CLI report identical
metrics.
