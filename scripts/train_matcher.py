"""Fine-tune the cross-encoder that decides BOL <-> Rate Con matches.

Reads the JSONL splits from ``build_training_set.py``, fine-tunes a small
transformer (DistilBERT by default) on the owner's GPU (RTX 5050) or CPU, and saves
the model to ``models/crossencoder``.

Usage:
    python scripts/build_training_set.py --count 500
    python scripts/train_matcher.py --epochs 3

Requires the ``[ml]`` extra (torch, transformers, sentence-transformers) on a
Python version PyTorch ships wheels for (3.11-3.13).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _load_jsonl(path: Path):
    from sentence_transformers import InputExample

    examples, raw = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            examples.append(InputExample(texts=[row["text_a"], row["text_b"]], label=float(row["label"])))
            raw.append(row)
    return examples, raw


def _evaluate(model, raw, threshold=0.5):
    texts = [[r["text_a"], r["text_b"]] for r in raw]
    scores = model.predict(texts)
    tp = fp = fn = tn = 0
    for s, r in zip(scores, raw):
        pred = int(s >= threshold)
        if pred and r["label"]:
            tp += 1
        elif pred and not r["label"]:
            fp += 1
        elif not pred and r["label"]:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--out", type=Path, default=Path("models/crossencoder"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_examples, _ = _load_jsonl(args.data / "train.jsonl")
    _, val_raw = _load_jsonl(args.data / "val.jsonl")
    _, test_raw = _load_jsonl(args.data / "test.jsonl")
    print(f"Loaded {len(train_examples)} train / {len(val_raw)} val / {len(test_raw)} test pairs")

    try:
        model = CrossEncoder(args.model, num_labels=1, device=device,
                             activation_fn=torch.nn.Sigmoid())
    except TypeError:  # sentence-transformers < 5 uses the old kwarg name
        model = CrossEncoder(args.model, num_labels=1, device=device,
                             default_activation_function=torch.nn.Sigmoid())
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup = max(1, int(len(train_loader) * args.epochs * 0.1))

    model.fit(train_dataloader=train_loader, epochs=args.epochs, warmup_steps=warmup, show_progress_bar=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    print(f"\nSaved fine-tuned cross-encoder to {args.out}")

    p, r, f1 = _evaluate(model, val_raw)
    print(f"Validation  P={p:.3f} R={r:.3f} F1={f1:.3f}")
    p, r, f1 = _evaluate(model, test_raw)
    print(f"Test        P={p:.3f} R={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
