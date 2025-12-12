"""Stage 1 — candidate generation (blocking) via a bi-encoder.

Embeds each document once with a sentence-transformer and, for every BOL, returns
only its top-k most similar Rate Cons. This shrinks the comparison space from
O(BOLs x RateCons) to O(BOLs x k) before the expensive cross-encoder runs.

``sentence-transformers`` (and torch) are imported lazily so importing this module
never forces the ML dependency on callers that only need the heuristic matcher.
"""

import logging
from typing import Callable

from matcher.linkage.serialize import serialize_doc
from matcher.models import ExtractedDocument

logger = logging.getLogger(__name__)

DEFAULT_BIENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Candidate generator: given a BOL and the full RC list, return [(rc_index, rc), ...].
CandidateFn = Callable[[ExtractedDocument, list[ExtractedDocument]], list[tuple[int, ExtractedDocument]]]


class BiEncoderBlocker:
    """Top-k nearest-neighbour candidate generator over Rate Cons."""

    def __init__(self, model_name: str = DEFAULT_BIENCODER, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy

        self.model = SentenceTransformer(model_name, device=device)

    def candidate_fn(self, rcs: list[ExtractedDocument], top_k: int = 5) -> CandidateFn:
        """Pre-embed the RC corpus once and return a closure for per-BOL retrieval."""
        from sentence_transformers import util  # lazy

        if not rcs:
            return lambda bol, _rcs: []

        rc_texts = [serialize_doc(rc) for rc in rcs]
        rc_emb = self.model.encode(rc_texts, convert_to_tensor=True, normalize_embeddings=True)
        k = min(top_k, len(rcs))

        def _fn(bol: ExtractedDocument, _rcs: list[ExtractedDocument]):
            q = self.model.encode(serialize_doc(bol), convert_to_tensor=True, normalize_embeddings=True)
            hits = util.semantic_search(q, rc_emb, top_k=k)[0]
            return [(h["corpus_id"], rcs[h["corpus_id"]]) for h in hits]

        return _fn


def _norm(s) -> str:
    return s.upper().replace("-", "").replace(" ", "").strip() if s else ""


def _struct_keys(doc: ExtractedDocument) -> list[tuple[str, str]]:
    """Structural blocking keys: shared ZIP / PO means a plausible same-load pair."""
    keys: list[tuple[str, str]] = []
    if doc.pickup_zip:
        keys.append(("pz", _norm(doc.pickup_zip)))
    if doc.delivery_zip:
        keys.append(("dz", _norm(doc.delivery_zip)))
    if doc.broker_po:
        keys.append(("po", _norm(doc.broker_po)))
    return keys


class HybridBlocker:
    """Structural blocking (shared ZIP/PO) UNION bi-encoder semantic top-k.

    Pure semantic similarity is a poor blocking key when records are near-duplicates
    (e.g. many loads on the same recurring lane): the true match can fall outside the
    top-k. Structural keys guarantee that same-lane / same-PO candidates are retrieved,
    while the semantic top-k catches matches whose structural fields were dropped or
    OCR-garbled. The cross-encoder then disambiguates the union.
    """

    def __init__(self, model_name: str = DEFAULT_BIENCODER, device: str | None = None):
        self._bi = BiEncoderBlocker(model_name, device=device)

    def candidate_fn(self, rcs: list[ExtractedDocument], top_k: int = 5) -> CandidateFn:
        if not rcs:
            return lambda bol, _rcs: []

        index: dict[tuple[str, str], list[int]] = {}
        for i, rc in enumerate(rcs):
            for key in _struct_keys(rc):
                index.setdefault(key, []).append(i)

        semantic = self._bi.candidate_fn(rcs, top_k=top_k)

        def _fn(bol: ExtractedDocument, _rcs: list[ExtractedDocument]):
            idx: set[int] = set()
            for key in _struct_keys(bol):
                idx.update(index.get(key, ()))
            idx.update(i for i, _ in semantic(bol, rcs))
            return [(i, rcs[i]) for i in sorted(idx)]

        return _fn


def all_candidates(bol: ExtractedDocument, rcs: list[ExtractedDocument]) -> list[tuple[int, ExtractedDocument]]:
    """No-blocking fallback: every RC is a candidate (used for benchmark control)."""
    return list(enumerate(rcs))
