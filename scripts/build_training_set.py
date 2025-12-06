"""Build a labeled entity-matching dataset for fine-tuning and benchmarking.

Emits JSONL train/val/test splits of serialized BOL/Rate-Con pairs:

    {"text_a": "[COL] load [VAL] ...", "text_b": "[COL] load [VAL] ...", "label": 1}

Usage:
    python scripts/build_training_set.py --count 500 --out data
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matcher.linkage.serialize import serialize_doc  # noqa: E402
from matcher.linkage.synth import build_pairs, split_pairs  # noqa: E402


def _dump(pairs, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps({
                "text_a": serialize_doc(p.bol),
                "text_b": serialize_doc(p.rc),
                "label": p.label,
            }) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=500, help="Number of loads to generate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.4, help="Extraction-noise rate [0,1].")
    ap.add_argument("--lanes", type=int, default=8,
                    help="Also mix in a recurring-lane corpus drawn from N lanes — its "
                         "near-duplicate negatives are what teach the model to disambiguate "
                         "same-lane loads (0 to disable).")
    ap.add_argument("--out", type=Path, default=Path("data"), help="Output directory.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(count=args.count, seed=args.seed, noise=args.noise)
    if args.lanes:
        # Recurring-lane half: different seed (no lane overlap with benchmarks) and
        # extra hard negatives, since in-lane disambiguation is the hard case.
        pairs += build_pairs(count=args.count, seed=args.seed + 1000, noise=args.noise,
                             hard_neg_per_pos=2, n_lanes=args.lanes)
    train, val, test = split_pairs(pairs, seed=args.seed)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = args.out / f"{name}.jsonl"
        _dump(split, path)
        pos = sum(p.label for p in split)
        print(f"  {name}: {len(split)} pairs ({pos} positive) -> {path}")

    print(f"\nDone — {len(pairs)} pairs from {args.count} loads written to {args.out}/")


if __name__ == "__main__":
    main()
