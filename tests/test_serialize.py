from datetime import date

from matcher.linkage.serialize import serialize_doc, serialize_pair
from matcher.models import DocType, ExtractedDocument, SourceType


def doc(**kw) -> ExtractedDocument:
    return ExtractedDocument(
        source_path=kw.pop("source_path", "/tmp/d.pdf"),
        doc_type=kw.pop("doc_type", DocType.BOL),
        source_type=SourceType.NATIVE_PDF,
        extraction_method="native",
        **kw,
    )


def test_serialize_is_deterministic():
    d = doc(load_number="LD1", broker="TQL", pickup_city="Chicago", pickup_state="IL")
    assert serialize_doc(d) == serialize_doc(d)


def test_serialize_has_all_columns_even_when_empty():
    d = doc()  # nothing populated
    text = serialize_doc(d)
    for col in ("load", "broker", "po", "pickup", "delivery", "weight", "rate", "carrier_mc"):
        assert f"[COL] {col} [VAL]" in text


def test_serialize_includes_values():
    d = doc(load_number="LD100003", broker="TQL", pickup_city="Atlanta",
            pickup_state="GA", pickup_zip="30301", pickup_date=date(2025, 10, 18),
            weight_lbs=32000.0, rate_amount=1100.0)
    text = serialize_doc(d)
    assert "LD100003" in text and "Atlanta GA 30301 2025-10-18" in text
    assert "32000" in text and "1100" in text  # floats formatted without trailing .0


def test_serialize_pair_returns_both_sides():
    a, b = serialize_pair(doc(load_number="A"), doc(load_number="B"))
    assert "A" in a and "B" in b
