"""Analytics layer tests — require the [bi] extra (duckdb, pandas)."""

from datetime import date

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pandas")

from matcher.analytics.load import load_to_duckdb, rows_from  # noqa: E402
from matcher.analytics.metrics import query_on  # noqa: E402
from matcher.models import DocType, ExtractedDocument, Match, SourceType  # noqa: E402


def bol(**kw):
    return ExtractedDocument(source_path=kw.pop("source_path", "/b.pdf"), doc_type=DocType.BOL,
                             source_type=SourceType.NATIVE_PDF, extraction_method="native", **kw)


def rc(**kw):
    return ExtractedDocument(source_path=kw.pop("source_path", "/r.pdf"), doc_type=DocType.RATE_CON,
                             source_type=SourceType.NATIVE_PDF, extraction_method="native", **kw)


def sample_match():
    return Match(
        bol=bol(source_path="/b1.pdf", load_number="LD1", broker="TQL",
                pickup_city="Chicago", pickup_state="IL", delivery_city="Dallas",
                delivery_state="TX", weight_lbs=20000.0, pickup_date=date(2025, 10, 1)),
        rate_con=rc(source_path="/r1.pdf", load_number="LD1", broker="TQL", rate_amount=1850.0,
                    delivery_city="Dallas", delivery_state="TX"),
        score=100.0, match_type="exact_load", reasons=["exact load number"],
    )


def test_match_row_coalesces_and_takes_rate_from_rate_con():
    rows = rows_from([sample_match()], [])
    assert len(rows) == 1
    r = rows[0]
    assert r["rate"] == 1850.0           # rate from the Rate Con
    assert r["lane"] == "Chicago, IL -> Dallas, TX"
    assert r["is_matched"] is True


def test_unmatched_doc_flagged_false():
    rows = rows_from([], [bol(load_number="ORPHAN")])
    assert rows[0]["is_matched"] is False
    assert rows[0]["match_type"] == "unmatched"


def test_load_to_duckdb_and_views(tmp_path):
    db = str(tmp_path / "a.duckdb")
    unmatched = [rc(source_path="/r9.pdf", load_number="ORPH", broker="COYOTE", rate_amount=500.0)]
    con = load_to_duckdb([sample_match()], unmatched, db)

    total = con.execute("SELECT COUNT(*) FROM loads").fetchone()[0]
    assert total == 2

    broker = query_on(con, "broker_scorecard").set_index("broker")
    assert broker.loc["TQL", "matched"] == 1
    assert broker.loc["TQL", "match_rate_pct"] == 100.0
    assert broker.loc["COYOTE", "matched"] == 0

    lanes = query_on(con, "lane_summary")
    assert (lanes["lane"] == "Chicago, IL -> Dallas, TX").any()

    exceptions = query_on(con, "exception_queue")
    assert (exceptions["load_number"] == "ORPH").any()
    con.close()
