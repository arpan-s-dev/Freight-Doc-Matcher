"""Deep entity-matching layer for pairing BOLs with Rate Confirmations.

This package upgrades the hand-tuned additive scorer in ``matcher.matcher`` to a
modern entity-matching pipeline (Ditto, Li et al., VLDB 2020):

    bi-encoder blocking (candidate generation)  ->  cross-encoder reranking (decision)

The classic additive scorer and a Fellegi-Sunter probabilistic model are kept in
``baselines`` purely as benchmark references. Heavy ML dependencies (torch,
transformers, sentence-transformers) are imported lazily inside ``blocker`` and
``crossencoder`` so the core ``matcher`` CLI runs without the ``[ml]`` extra.
"""

from matcher.linkage.serialize import serialize_doc, serialize_pair

__all__ = ["serialize_doc", "serialize_pair"]
