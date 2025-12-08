from datetime import date

from matcher.linkage.baselines import FellegiSunter, comparison_vector
from matcher.linkage.synth import build_pairs, split_pairs
from matcher.models import DocType, ExtractedDocument, SourceType


def bol(**kw):
    return ExtractedDocument(source_path="/tmp/b.pdf", doc_type=DocType.BOL,
                             source_type=SourceType.NATIVE_PDF, extraction_method="native", **kw)


def rc(**kw):
    return ExtractedDocument(source_path="/tmp/r.pdf", doc_type=DocType.RATE_CON,
                             source_type=SourceType.NATIVE_PDF, extraction_method="native", **kw)


def test_comparison_vector_exact_load():
    cmp = comparison_vector(bol(load_number="LD-100"), rc(load_number="ld100"))
    assert cmp["load_number"] is True


def test_comparison_vector_date_within_one_day():
    cmp = comparison_vector(
        bol(pickup_date=date(2025, 10, 1)), rc(pickup_date=date(2025, 10, 2)))
    assert cmp["pickup_date"] is True
    cmp = comparison_vector(
        bol(pickup_date=date(2025, 10, 1)), rc(pickup_date=date(2025, 10, 5)))
    assert cmp["pickup_date"] is False


def test_fellegi_sunter_weights_make_agreement_positive():
    # On labeled synthetic data, agreeing on load_number should carry a large
    # positive weight (m >> u), and m/u stay strictly inside (0, 1) via smoothing.
    pairs = build_pairs(count=120, seed=1)
    fs = FellegiSunter().fit(pairs)
    assert 0.0 < fs.u["load_number"] < fs.m["load_number"] < 1.0
    assert fs.weight("load_number", True) > 0
    assert fs.weight("load_number", False) < 0


def test_fellegi_sunter_separates_matches_from_nonmatches():
    pairs = build_pairs(count=200, seed=7)
    train, val, test = split_pairs(pairs, seed=7)
    fs = FellegiSunter().fit(train).calibrate_threshold(val)
    correct = sum(1 for p in test if fs.predict(p.bol, p.rc) == p.label)
    assert correct / len(test) > 0.9
