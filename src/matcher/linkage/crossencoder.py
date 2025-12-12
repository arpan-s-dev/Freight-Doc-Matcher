"""Stage 2 — the fine-tuned cross-encoder that decides BOL <-> Rate Con matches.

Wraps ``sentence_transformers.CrossEncoder`` (a transformer such as DistilBERT that
reads both serialized records jointly and outputs a match probability). torch and
sentence-transformers are imported lazily so the module is import-safe without the
``[ml]`` extra.
"""

import logging
from typing import Callable

from matcher.linkage.serialize import serialize_doc
from matcher.models import ExtractedDocument

logger = logging.getLogger(__name__)

DEFAULT_CROSSENCODER = "distilbert-base-uncased"

# A scorer maps a (bol, rc) pair to a match probability in [0, 1].
Scorer = Callable[[ExtractedDocument, ExtractedDocument], float]


def detect_device() -> str:
    """Return 'cuda' when a GPU is available, else 'cpu'."""
    try:
        import torch  # lazy
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class CrossEncoderMatcher:
    """Inference wrapper around a (fine-tuned) CrossEncoder."""

    def __init__(self, model_path: str, device: str | None = None):
        from sentence_transformers import CrossEncoder  # lazy

        self.model = CrossEncoder(model_path, device=device or detect_device())

    def score(self, bol: ExtractedDocument, rc: ExtractedDocument) -> float:
        return float(self.model.predict([[serialize_doc(bol), serialize_doc(rc)]])[0])

    def score_batch(self, pairs: list[tuple[ExtractedDocument, ExtractedDocument]]) -> list[float]:
        if not pairs:
            return []
        texts = [[serialize_doc(b), serialize_doc(r)] for b, r in pairs]
        return [float(s) for s in self.model.predict(texts)]

    def scorer(self) -> Scorer:
        return self.score
