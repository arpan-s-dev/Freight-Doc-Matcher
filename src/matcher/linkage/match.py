"""Glue the entity-matching pipeline into the matcher's existing contract.

``match_documents_ml`` returns the same ``(list[Match], list[ExtractedDocument])``
shape as ``matcher.matcher.match_documents``, so downstream code (organize,
spreadsheet, analytics) is unchanged. It accepts an injected ``scorer`` and
``candidate_fn`` so unit tests can run with lightweight stubs (no torch/GPU), while
``build_ml_matcher`` wires the real bi-encoder + cross-encoder for production use.
"""

import logging
from typing import Optional

from matcher.linkage.blocker import CandidateFn, all_candidates
from matcher.linkage.crossencoder import Scorer
from matcher.models import DocType, ExtractedDocument, Match

logger = logging.getLogger(__name__)


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.upper().replace("-", "").replace(" ", "").strip()


def _match_type(prob: float, bol: ExtractedDocument, rc: ExtractedDocument, auto: float) -> str:
    bl, rl = _norm(bol.load_number), _norm(rc.load_number)
    if bl and rl and bl == rl:
        return "exact_load"
    return "fuzzy" if prob >= auto else "manual_review"


def match_documents_ml(
    docs: list[ExtractedDocument],
    scorer: Scorer,
    candidate_fn: Optional[CandidateFn] = None,
    auto_threshold: float = 0.70,
    review_threshold: float = 0.40,
) -> tuple[list[Match], list[ExtractedDocument]]:
    """Block -> rerank -> pick the best Rate Con per BOL.

    A BOL with no candidate above ``review_threshold`` stays unmatched. Each Rate
    Con is consumed at most once (greedy, highest-confidence BOL wins it first via
    document order — mirroring the heuristic matcher's one-RC-per-match rule).
    """
    bols = [d for d in docs if d.doc_type == DocType.BOL]
    rcs = [d for d in docs if d.doc_type == DocType.RATE_CON]
    candidate_fn = candidate_fn or all_candidates

    matched_rc_idx: set[int] = set()
    matches: list[Match] = []

    for bol in bols:
        candidates = candidate_fn(bol, rcs)
        scored = sorted(
            ((i, scorer(bol, rcs[i])) for i, _ in candidates if i not in matched_rc_idx),
            key=lambda x: x[1],
            reverse=True,
        )
        if not scored or scored[0][1] < review_threshold:
            continue

        i, prob = scored[0]
        rc = rcs[i]
        mtype = _match_type(prob, bol, rc, auto_threshold)
        matches.append(Match(
            bol=bol, rate_con=rc, score=round(prob * 100, 1),
            match_type=mtype, reasons=[f"cross-encoder p={prob:.2f}"],
        ))
        matched_rc_idx.add(i)

    matched_bol_paths = {m.bol.source_path for m in matches}
    unmatched = (
        [b for b in bols if b.source_path not in matched_bol_paths]
        + [rcs[i] for i in range(len(rcs)) if i not in matched_rc_idx]
    )
    return matches, unmatched


def build_ml_matcher(
    docs: list[ExtractedDocument],
    model_path: str = "models/crossencoder",
    top_k: int = 10,
    blocking: str = "hybrid",
):
    """Construct the real (scorer, candidate_fn) from fine-tuned models.

    ``blocking`` is one of ``hybrid`` (structural keys + semantic top-k; recommended),
    ``semantic`` (bi-encoder top-k only), or ``none`` (exhaustive). Imports the ML
    stack lazily; raises if torch / a trained model is unavailable so the CLI can fall
    back to the heuristic matcher.
    """
    from matcher.linkage.crossencoder import CrossEncoderMatcher

    rcs = [d for d in docs if d.doc_type == DocType.RATE_CON]
    matcher = CrossEncoderMatcher(model_path)

    candidate_fn = all_candidates
    if blocking != "none" and rcs:
        if blocking == "hybrid":
            from matcher.linkage.blocker import HybridBlocker
            candidate_fn = HybridBlocker().candidate_fn(rcs, top_k=top_k)
        elif blocking == "semantic":
            from matcher.linkage.blocker import BiEncoderBlocker
            candidate_fn = BiEncoderBlocker().candidate_fn(rcs, top_k=top_k)
        else:
            raise ValueError(f"unknown blocking mode: {blocking}")

    return matcher.scorer(), candidate_fn
