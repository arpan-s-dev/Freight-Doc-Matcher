from datetime import date
from pathlib import Path

import pytest

from matcher.matcher import match_documents, score_pair
from matcher.models import DocType, ExtractedDocument, SourceType


def bol(**kw) -> ExtractedDocument:
    return ExtractedDocument(
        source_path=kw.pop("source_path", "/tmp/bol.pdf"),
        doc_type=DocType.BOL,
        source_type=SourceType.NATIVE_PDF,
        extraction_method="native",
        **kw,
    )


def rc(**kw) -> ExtractedDocument:
    return ExtractedDocument(
        source_path=kw.pop("source_path", "/tmp/rc.pdf"),
        doc_type=DocType.RATE_CON,
        source_type=SourceType.NATIVE_PDF,
        extraction_method="native",
        **kw,
    )


# ── score_pair ───────────────────────────────────────────────────────────────

def test_exact_load_number():
    s, reasons = score_pair(bol(load_number="LOAD12345"), rc(load_number="LOAD12345"))
    assert s == 100.0
    assert "exact load number" in reasons


def test_exact_load_normalized_dashes():
    s, _ = score_pair(bol(load_number="LOAD-123-45"), rc(load_number="load12345"))
    assert s == 100.0


def test_fuzzy_match_via_zips_and_dates():
    b = bol(pickup_zip="92590", delivery_zip="95376",
            pickup_date=date(2025, 10, 9), delivery_date=date(2025, 10, 10),
            pickup_city="Temecula", delivery_city="Tracy", broker="TQL")
    r = rc(pickup_zip="92590", delivery_zip="95376",
           pickup_date=date(2025, 10, 9), delivery_date=date(2025, 10, 10),
           pickup_city="Temecula", delivery_city="Tracy", broker="TQL")
    s, _ = score_pair(b, r)
    assert s >= 70


def test_no_match_different_zips():
    s, _ = score_pair(bol(pickup_zip="10001", delivery_zip="20001"),
                      rc(pickup_zip="90210", delivery_zip="60601"))
    assert s < 50


def test_date_within_one_day():
    b = bol(pickup_date=date(2025, 10, 9))
    r = rc(pickup_date=date(2025, 10, 10))
    s, reasons = score_pair(b, r)
    assert "pickup date" in reasons


def test_weight_within_5pct():
    b = bol(weight_lbs=20000.0)
    r = rc(weight_lbs=20500.0)
    _, reasons = score_pair(b, r)
    assert "weight match" in reasons


def test_weight_outside_5pct():
    b = bol(weight_lbs=20000.0)
    r = rc(weight_lbs=22000.0)
    _, reasons = score_pair(b, r)
    assert "weight match" not in reasons


# ── match_documents ──────────────────────────────────────────────────────────

def test_exact_match_end_to_end():
    b = bol(source_path="/tmp/b1.pdf", load_number="LD001")
    r = rc(source_path="/tmp/r1.pdf", load_number="LD001")
    matches, unmatched = match_documents([b, r])
    assert len(matches) == 1
    assert matches[0].match_type == "exact_load"
    assert len(unmatched) == 0


def test_unmatched_docs():
    b = bol(source_path="/tmp/b_lone.pdf", load_number="LD999")
    r = rc(source_path="/tmp/r_lone.pdf", load_number="LD888")
    matches, unmatched = match_documents([b, r])
    assert len(matches) == 0
    assert len(unmatched) == 2


def test_tie_flagged_for_review():
    b = bol(source_path="/tmp/b_tie.pdf", pickup_zip="12345", delivery_zip="67890")
    r1 = rc(source_path="/tmp/r_tie1.pdf", pickup_zip="12345", delivery_zip="67890")
    r2 = rc(source_path="/tmp/r_tie2.pdf", pickup_zip="12345", delivery_zip="67890")
    matches, _ = match_documents([b, r1, r2])
    assert all(m.match_type == "manual_review" for m in matches)


def test_below_50_threshold_not_matched():
    b = bol(source_path="/tmp/b_low.pdf")
    r = rc(source_path="/tmp/r_low.pdf")
    matches, unmatched = match_documents([b, r])
    assert len(matches) == 0
    assert len(unmatched) == 2


def test_manual_review_range():
    b = bol(source_path="/tmp/b_50.pdf",
            pickup_zip="11111", delivery_zip="22222",
            pickup_date=date(2025, 10, 1))
    r = rc(source_path="/tmp/r_50.pdf",
            pickup_zip="11111", delivery_zip="22222",
            pickup_date=date(2025, 10, 1))
    matches, _ = match_documents([b, r])
    # score = 20+20+15 = 55 → manual_review
    assert len(matches) == 1
    assert matches[0].match_type == "manual_review"
