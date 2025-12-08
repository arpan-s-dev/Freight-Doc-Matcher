"""Benchmark baselines: the existing additive heuristic + Fellegi-Sunter.

These exist only so the efficiency write-up can show a progression
(hand-tuned heuristic -> classic probabilistic linkage -> fine-tuned transformer)
with rising F1. The heuristic is imported unchanged from ``matcher.matcher``.

Fellegi-Sunter (1969) is the classic record-linkage model: for each comparison
field it estimates ``m = P(agree | match)`` and ``u = P(agree | non-match)`` and
sums the log-likelihood-ratio weights into a match score. Here we estimate m/u
*supervised* from labeled pairs (we have ground-truth labels from the synthetic
generator).
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from matcher.matcher import score_pair as heuristic_score  # noqa: F401 (re-exported)
from matcher.models import ExtractedDocument


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.upper().replace("-", "").replace(" ", "").strip()


def comparison_vector(bol: ExtractedDocument, rc: ExtractedDocument) -> dict[str, bool]:
    """Per-field agreement flags, mirroring the heuristic's comparison logic."""
    cmp: dict[str, bool] = {}
    bl, rl = _norm(bol.load_number), _norm(rc.load_number)
    cmp["load_number"] = bool(bl and rl and bl == rl)

    bp, rp = _norm(bol.broker_po), _norm(rc.broker_po)
    cmp["broker_po"] = bool(bp and rp and bp == rp)

    cmp["pickup_zip"] = bool(bol.pickup_zip and rc.pickup_zip and bol.pickup_zip == rc.pickup_zip)
    cmp["delivery_zip"] = bool(bol.delivery_zip and rc.delivery_zip and bol.delivery_zip == rc.delivery_zip)

    cmp["pickup_date"] = bool(
        bol.pickup_date and rc.pickup_date and abs((bol.pickup_date - rc.pickup_date).days) <= 1
    )
    cmp["delivery_date"] = bool(
        bol.delivery_date and rc.delivery_date and abs((bol.delivery_date - rc.delivery_date).days) <= 1
    )

    cmp["weight"] = bool(
        bol.weight_lbs and rc.weight_lbs and bol.weight_lbs > 0
        and abs(bol.weight_lbs - rc.weight_lbs) / bol.weight_lbs <= 0.05
    )

    cmp["pickup_city"] = bool(
        bol.pickup_city and rc.pickup_city
        and fuzz.ratio(bol.pickup_city.lower(), rc.pickup_city.lower()) > 85
    )
    cmp["delivery_city"] = bool(
        bol.delivery_city and rc.delivery_city
        and fuzz.ratio(bol.delivery_city.lower(), rc.delivery_city.lower()) > 85
    )
    cmp["broker"] = bool(bol.broker and rc.broker and bol.broker == rc.broker)
    return cmp


_FIELDS = [
    "load_number", "broker_po", "pickup_zip", "delivery_zip",
    "pickup_date", "delivery_date", "weight", "pickup_city", "delivery_city", "broker",
]


@dataclass
class FellegiSunter:
    """Supervised Fellegi-Sunter probabilistic record-linkage scorer."""

    m: dict[str, float] = field(default_factory=dict)
    u: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0

    def fit(self, pairs) -> "FellegiSunter":
        """Estimate m/u from labeled pairs.

        ``pairs`` is an iterable of objects with ``.bol``, ``.rc``, ``.label``.
        Laplace smoothing avoids zero/one probabilities (log of 0).
        """
        agree_match = {f: 0 for f in _FIELDS}
        agree_non = {f: 0 for f in _FIELDS}
        n_match = n_non = 0
        for p in pairs:
            cmp = comparison_vector(p.bol, p.rc)
            if p.label == 1:
                n_match += 1
                for f in _FIELDS:
                    agree_match[f] += int(cmp[f])
            else:
                n_non += 1
                for f in _FIELDS:
                    agree_non[f] += int(cmp[f])
        for f in _FIELDS:
            self.m[f] = (agree_match[f] + 1) / (n_match + 2)
            self.u[f] = (agree_non[f] + 1) / (n_non + 2)
        return self

    def weight(self, field_name: str, agrees: bool) -> float:
        m, u = self.m[field_name], self.u[field_name]
        if agrees:
            return math.log2(m / u)
        return math.log2((1 - m) / (1 - u))

    def score(self, bol: ExtractedDocument, rc: ExtractedDocument) -> float:
        cmp = comparison_vector(bol, rc)
        return sum(self.weight(f, cmp[f]) for f in _FIELDS)

    def predict(self, bol: ExtractedDocument, rc: ExtractedDocument) -> int:
        return int(self.score(bol, rc) >= self.threshold)

    def calibrate_threshold(self, val_pairs) -> "FellegiSunter":
        """Pick the score threshold that maximizes F1 on a validation set."""
        scored = [(self.score(p.bol, p.rc), p.label) for p in val_pairs]
        if not scored:
            return self
        candidates = sorted({s for s, _ in scored})
        best_f1, best_t = -1.0, 0.0
        for t in candidates:
            tp = sum(1 for s, y in scored if s >= t and y == 1)
            fp = sum(1 for s, y in scored if s >= t and y == 0)
            fn = sum(1 for s, y in scored if s < t and y == 1)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = best_t
        return self
