"""Load matcher output into a DuckDB ``loads`` table (one row per load/document).

A matched load coalesces BOL/Rate-Con fields the same way the spreadsheet does
(`bol.x or rc.x`), with the rate taken from the Rate Con. Unmatched documents are
inserted with ``is_matched = False`` so exception analytics work.
"""

from pathlib import Path
from typing import Optional

from matcher.models import ExtractedDocument, Match

DEFAULT_DB = "output/analytics.duckdb"

_COLUMNS = [
    "load_number", "broker", "lane",
    "pickup_city", "pickup_state", "pickup_zip", "pickup_date",
    "delivery_city", "delivery_state", "delivery_zip", "delivery_date",
    "weight_lbs", "rate", "match_score", "match_type", "is_matched",
    "doc_type", "extraction_method", "confidence", "bol_path", "rc_path",
]


def _coalesce(a, b):
    return a if a is not None else b


def _lane(pickup_city, pickup_state, delivery_city, delivery_state) -> Optional[str]:
    if pickup_city and delivery_city:
        o = f"{pickup_city}, {pickup_state}" if pickup_state else pickup_city
        d = f"{delivery_city}, {delivery_state}" if delivery_state else delivery_city
        return f"{o} -> {d}"
    return None


def _match_row(m: Match) -> dict:
    bol, rc = m.bol, m.rate_con
    pickup_city = _coalesce(bol.pickup_city, rc.pickup_city)
    pickup_state = _coalesce(bol.pickup_state, rc.pickup_state)
    delivery_city = _coalesce(bol.delivery_city, rc.delivery_city)
    delivery_state = _coalesce(bol.delivery_state, rc.delivery_state)
    return {
        "load_number": _coalesce(bol.load_number, rc.load_number),
        "broker": _coalesce(bol.broker, rc.broker),
        "lane": _lane(pickup_city, pickup_state, delivery_city, delivery_state),
        "pickup_city": pickup_city, "pickup_state": pickup_state,
        "pickup_zip": _coalesce(bol.pickup_zip, rc.pickup_zip),
        "pickup_date": _coalesce(bol.pickup_date, rc.pickup_date),
        "delivery_city": delivery_city, "delivery_state": delivery_state,
        "delivery_zip": _coalesce(bol.delivery_zip, rc.delivery_zip),
        "delivery_date": _coalesce(bol.delivery_date, rc.delivery_date),
        "weight_lbs": _coalesce(bol.weight_lbs, rc.weight_lbs),
        "rate": _coalesce(rc.rate_amount, bol.rate_amount),
        "match_score": m.score, "match_type": m.match_type, "is_matched": True,
        "doc_type": "MATCH", "extraction_method": bol.extraction_method,
        "confidence": min(bol.confidence, rc.confidence),
        "bol_path": str(bol.source_path), "rc_path": str(rc.source_path),
    }


def _doc_row(d: ExtractedDocument) -> dict:
    return {
        "load_number": d.load_number, "broker": d.broker,
        "lane": _lane(d.pickup_city, d.pickup_state, d.delivery_city, d.delivery_state),
        "pickup_city": d.pickup_city, "pickup_state": d.pickup_state,
        "pickup_zip": d.pickup_zip, "pickup_date": d.pickup_date,
        "delivery_city": d.delivery_city, "delivery_state": d.delivery_state,
        "delivery_zip": d.delivery_zip, "delivery_date": d.delivery_date,
        "weight_lbs": d.weight_lbs, "rate": d.rate_amount,
        "match_score": 0.0, "match_type": "unmatched", "is_matched": False,
        "doc_type": d.doc_type.value, "extraction_method": d.extraction_method,
        "confidence": d.confidence,
        "bol_path": str(d.source_path) if d.doc_type.value == "BOL" else None,
        "rc_path": str(d.source_path) if d.doc_type.value == "RATE_CON" else None,
    }


def rows_from(matches: list[Match], unmatched: list[ExtractedDocument]) -> list[dict]:
    return [_match_row(m) for m in matches] + [_doc_row(d) for d in unmatched]


def load_to_duckdb(matches: list[Match], unmatched: list[ExtractedDocument],
                   db_path: str = DEFAULT_DB):
    """Replace the ``loads`` table in ``db_path`` with the current run's output."""
    import duckdb
    import pandas as pd

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows_from(matches, unmatched), columns=_COLUMNS)

    con = duckdb.connect(db_path)
    con.register("incoming", df)
    con.execute("CREATE OR REPLACE TABLE loads AS SELECT * FROM incoming")
    con.unregister("incoming")
    return con
