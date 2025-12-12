"""Blocker test — skipped unless the ML extra (sentence-transformers) is installed.

Verifies that top-k retrieval surfaces the true Rate Con and shrinks the candidate
set. Requires downloading a small model, so it is opt-in via the dependency.
"""

import pytest

pytest.importorskip("sentence_transformers")

from matcher.linkage.blocker import BiEncoderBlocker  # noqa: E402
from matcher.linkage.synth import generate_loads  # noqa: E402
from matcher.linkage.synth import _bol_from_load, _rc_from_load  # noqa: E402
import random  # noqa: E402


@pytest.mark.slow
def test_topk_retrieves_true_rc_and_shrinks_candidates():
    loads = generate_loads(20, seed=3)
    bols = [_bol_from_load(ld) for ld in loads]
    rng = random.Random(3)
    rcs = [_rc_from_load(ld, rng, noise=0.0) for ld in loads]

    blocker = BiEncoderBlocker()
    fn = blocker.candidate_fn(rcs, top_k=5)

    hits = 0
    for i, bol in enumerate(bols):
        cand_idx = [idx for idx, _ in fn(bol, rcs)]
        assert len(cand_idx) <= 5  # far fewer than the 20 RCs
        if i in cand_idx:
            hits += 1
    assert hits / len(bols) >= 0.9  # true RC almost always in the top-5
