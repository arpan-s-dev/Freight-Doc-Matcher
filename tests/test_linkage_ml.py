"""Tests for the ML matching orchestration using a stub scorer (no torch/GPU).

These verify the wiring (candidate selection, thresholds, one-RC-per-match, return
shape) independently of the actual transformer, which is exercised by training.
"""

from matcher.linkage.match import match_documents_ml
from matcher.models import DocType, ExtractedDocument, SourceType


def bol(path, **kw):
    return ExtractedDocument(source_path=path, doc_type=DocType.BOL,
                             source_type=SourceType.NATIVE_PDF, extraction_method="native", **kw)


def rc(path, **kw):
    return ExtractedDocument(source_path=path, doc_type=DocType.RATE_CON,
                             source_type=SourceType.NATIVE_PDF, extraction_method="native", **kw)


def make_scorer(table):
    """Scorer returning a fixed probability keyed by (bol.load_number, rc.load_number)."""
    return lambda b, r: table.get((b.load_number, r.load_number), 0.0)


def test_picks_highest_scoring_rc():
    docs = [
        bol("/b1.pdf", load_number="A"),
        rc("/r1.pdf", load_number="A"),
        rc("/r2.pdf", load_number="X"),
    ]
    scorer = make_scorer({("A", "A"): 0.95, ("A", "X"): 0.2})
    matches, unmatched = match_documents_ml(docs, scorer)
    assert len(matches) == 1
    assert matches[0].rate_con.load_number == "A"
    assert matches[0].match_type == "exact_load"  # normalized load numbers equal
    assert {u.load_number for u in unmatched} == {"X"}


def test_below_review_threshold_is_unmatched():
    docs = [bol("/b1.pdf", load_number="A"), rc("/r1.pdf", load_number="B")]
    matches, unmatched = match_documents_ml(docs, make_scorer({("A", "B"): 0.1}))
    assert matches == []
    assert len(unmatched) == 2


def test_mid_confidence_is_manual_review():
    docs = [bol("/b1.pdf", load_number="A"), rc("/r1.pdf", load_number="B")]
    matches, _ = match_documents_ml(docs, make_scorer({("A", "B"): 0.55}),
                                    auto_threshold=0.7, review_threshold=0.4)
    assert len(matches) == 1
    assert matches[0].match_type == "manual_review"


def test_rc_consumed_once():
    docs = [
        bol("/b1.pdf", load_number="A"),
        bol("/b2.pdf", load_number="A2"),
        rc("/r1.pdf", load_number="A"),
    ]
    # Both BOLs love the same RC; first BOL in document order wins it.
    scorer = make_scorer({("A", "A"): 0.9, ("A2", "A"): 0.95})
    matches, unmatched = match_documents_ml(docs, scorer)
    assert len(matches) == 1
    assert matches[0].bol.load_number == "A"
    assert any(u.load_number == "A2" for u in unmatched)


def test_candidate_fn_limits_comparisons():
    calls = []
    docs = [bol("/b1.pdf", load_number="A"),
            rc("/r1.pdf", load_number="A"), rc("/r2.pdf", load_number="X")]

    def scorer(b, r):
        calls.append((b.load_number, r.load_number))
        return 0.9 if r.load_number == "A" else 0.1

    # Blocker returns only the first RC as a candidate.
    candidate_fn = lambda b, rcs: [(0, rcs[0])]
    matches, _ = match_documents_ml(docs, scorer, candidate_fn=candidate_fn)
    assert len(matches) == 1
    assert calls == [("A", "A")]  # the X candidate was never scored
