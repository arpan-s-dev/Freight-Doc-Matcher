"""Benchmark the matching strategies on labeled synthetic data.

Reports, per strategy, end-to-end matching precision / recall / F1, the number of
pairwise comparisons performed, and wall-clock time:

    heuristic        -> the existing additive scorer (matcher.matcher)
    fellegi-sunter   -> classic probabilistic record linkage (fit on a train split)
    ml (cross-enc)   -> the fine-tuned transformer, with bi-encoder blocking
                        (only if torch + a trained model are available)

Ground truth: a BOL and Rate Con share a load id => true match.

Usage:
    python scripts/benchmark_matching.py --count 300
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matcher.linkage.baselines import FellegiSunter, heuristic_score  # noqa: E402
from matcher.linkage.synth import (  # noqa: E402
    _bol_from_load, _rc_from_load, build_pairs, generate_loads, split_pairs,
)
import random  # noqa: E402


def _true_load(path: Path) -> str:
    # Synthetic paths look like .../BOL_LD100003.pdf or .../RATECON_LD100003.pdf
    return path.stem.split("_", 1)[1]


def _greedy_match(bols, rcs, score_fn, threshold):
    """Generic one-RC-per-BOL greedy matcher. Returns (pairs, n_comparisons)."""
    used: set[int] = set()
    pairs = []
    comparisons = 0
    for bol in bols:
        best_i, best_s = None, threshold
        for i, rc in enumerate(rcs):
            if i in used:
                continue
            comparisons += 1
            s = score_fn(bol, rc)
            if s >= best_s:
                best_i, best_s = i, s
        if best_i is not None:
            pairs.append((_true_load(bol.source_path), _true_load(rcs[best_i].source_path)))
            used.add(best_i)
    return pairs, comparisons


def _prf(pred_pairs, n_true):
    tp = sum(1 for a, b in pred_pairs if a == b)
    fp = len(pred_pairs) - tp
    fn = n_true - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _row(name, prec, rec, f1, comps, secs):
    return f"| {name:<22} | {prec:6.3f} | {rec:6.3f} | {f1:6.3f} | {comps:>10,} | {secs:7.3f} |"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--lanes", type=int, default=None,
                    help="Draw from N recurring lanes (the hard, ambiguous scenario).")
    ap.add_argument("--date-window", type=int, default=120)
    ap.add_argument("--model", default="models/crossencoder")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--blocking", choices=["hybrid", "semantic", "none"], default="hybrid",
                    help="ML candidate-generation strategy.")
    args = ap.parse_args()

    # Document-level test corpus: every load contributes one BOL and one RC.
    loads = generate_loads(args.count, args.seed, n_lanes=args.lanes, date_window=args.date_window)
    rng = random.Random(args.seed)
    bols = [_bol_from_load(ld) for ld in loads]
    rcs = [_rc_from_load(ld, rng, args.noise) for ld in loads]
    n_true = len(loads)
    all_pairs = len(bols) * len(rcs)

    print(f"\nCorpus: {len(bols)} BOLs x {len(rcs)} Rate Cons = {all_pairs:,} possible pairs\n")
    header = f"| {'strategy':<22} | {'prec':>6} | {'recall':>6} | {'F1':>6} | {'compares':>10} | {'secs':>7} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")

    # 1) Heuristic additive scorer (normalize 0-100 -> 0-1; >=0.5 == its review bar).
    t = time.perf_counter()
    pairs, comps = _greedy_match(bols, rcs, lambda b, r: heuristic_score(b, r)[0] / 100.0, 0.5)
    print(_row("heuristic", *_prf(pairs, n_true), comps, time.perf_counter() - t))

    # 2) Fellegi-Sunter (fit on an independent labeled split, calibrate threshold).
    train, val, _ = split_pairs(
        build_pairs(args.count, args.seed + 1, args.noise, n_lanes=args.lanes), seed=args.seed)
    fs = FellegiSunter().fit(train).calibrate_threshold(val)
    t = time.perf_counter()
    pairs, comps = _greedy_match(bols, rcs, fs.score, fs.threshold)
    print(_row("fellegi-sunter", *_prf(pairs, n_true), comps, time.perf_counter() - t))

    # 3) Fine-tuned cross-encoder + bi-encoder blocking (optional).
    try:
        from matcher.linkage.blocker import BiEncoderBlocker, HybridBlocker, all_candidates
        from matcher.linkage.crossencoder import CrossEncoderMatcher

        ce = CrossEncoderMatcher(args.model)
        if args.blocking == "hybrid":
            block = HybridBlocker().candidate_fn(rcs, top_k=args.top_k)
            label = f"ml (hybrid k={args.top_k})"
        elif args.blocking == "semantic":
            block = BiEncoderBlocker().candidate_fn(rcs, top_k=args.top_k)
            label = f"ml (semantic k={args.top_k})"
        else:
            block = lambda b, r: all_candidates(b, r)
            label = "ml (no blocking)"
        t = time.perf_counter()
        used, pairs, comps = set(), [], 0
        for bol in bols:
            cands = [(i, rc) for i, rc in block(bol, rcs) if i not in used]
            comps += len(cands)
            scored = sorted(((i, ce.score(bol, rc)) for i, rc in cands), key=lambda x: x[1], reverse=True)
            if scored and scored[0][1] >= 0.5:
                i = scored[0][0]
                pairs.append((_true_load(bol.source_path), _true_load(rcs[i].source_path)))
                used.add(i)
        print(_row(label, *_prf(pairs, n_true), comps, time.perf_counter() - t))
    except Exception as e:
        print(f"| {'ml (cross-encoder)':<22} | skipped: {e}")

    print("\n(ml row appears once `[ml]` deps + a trained model in models/crossencoder exist.)")


if __name__ == "__main__":
    main()
