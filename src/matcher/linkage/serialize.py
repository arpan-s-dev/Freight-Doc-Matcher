"""Serialize an ExtractedDocument into a Ditto-style tagged string.

Entity-matching transformers (Ditto, VLDB 2020) operate on text, so each document
is flattened into a ``[COL] <attr> [VAL] <value>`` sequence. The same serialization
feeds both the bi-encoder (blocking) and the cross-encoder (decision), and is used
when building the training set so train/inference text are identical.
"""

from typing import Optional

from matcher.models import ExtractedDocument

# Column order is fixed so the serialization is deterministic across runs.
_COL = "[COL]"
_VAL = "[VAL]"


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        # Drop trailing .0 on whole numbers (weights, rates) for cleaner tokens.
        return f"{value:g}"
    return str(value).strip()


def _loc(city: Optional[str], state: Optional[str], zip_code: Optional[str], date) -> str:
    parts = [_fmt(city), _fmt(state), _fmt(zip_code), _fmt(date)]
    return " ".join(p for p in parts if p)


def serialize_doc(doc: ExtractedDocument) -> str:
    """Flatten a document into a single tagged string.

    Empty attributes are emitted with an empty value (rather than dropped) so the
    column structure is stable regardless of which fields were extracted.
    """
    fields = [
        ("load", _fmt(doc.load_number)),
        ("broker", _fmt(doc.broker)),
        ("po", _fmt(doc.broker_po)),
        ("pickup", _loc(doc.pickup_city, doc.pickup_state, doc.pickup_zip, doc.pickup_date)),
        ("delivery", _loc(doc.delivery_city, doc.delivery_state, doc.delivery_zip, doc.delivery_date)),
        ("weight", _fmt(doc.weight_lbs)),
        ("rate", _fmt(doc.rate_amount)),
        ("carrier_mc", _fmt(doc.carrier_mc)),
    ]
    return " ".join(f"{_COL} {name} {_VAL} {value}" for name, value in fields)


def serialize_pair(a: ExtractedDocument, b: ExtractedDocument) -> tuple[str, str]:
    """Serialize both sides of a candidate pair (for cross-encoder input)."""
    return serialize_doc(a), serialize_doc(b)
