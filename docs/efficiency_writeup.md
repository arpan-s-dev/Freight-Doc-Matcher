# From Hand-Tuned Rules to a Fine-Tuned Transformer: Making Freight Document Matching Robust

*How we replaced a brittle 100-point scoring heuristic with a deep entity-matching
pipeline — and measured the difference.*

## The problem

A freight dispatcher receives a daily pile of PDFs: **Bills of Lading (BOLs)** and
**Rate Confirmations** from a dozen brokers, each in its own format. The job is to
pair each BOL with its matching Rate Con so the load can be billed and tracked.
Freight Doc Matcher already extracts the fields (load #, pickup/delivery ZIP, dates,
weight, rate, broker) from each document; the question is **which BOL goes with which
Rate Con.**

This is a textbook **entity-matching** (a.k.a. record-linkage) problem: given two
records, decide whether they describe the same real-world entity.

## The original approach: a hand-tuned additive scorer

The first version scored every (BOL, Rate Con) pair with a 100-point rulebook:

```
exact load number  -> 100 (short-circuit)
broker PO match     -> +40
pickup ZIP match    -> +20
delivery ZIP match  -> +20
pickup date ±1 day  -> +15
delivery date ±1day -> +15
weight within 5%    -> +10
fuzzy city match    -> +5 each
same broker         -> +5
```
≥70 auto-matches, 50–69 goes to manual review. It works, but it has two weaknesses:

1. **The weights are guesses.** Why is a ZIP match worth 20 and a weight match worth
   10? Nobody learned those numbers from data.
2. **It leans on the load number.** When OCR garbles the load number (`LD100003` →
   `LD1O0003`) or a broker omits it, the scorer falls back to ZIP/date/weight — which
   is fine *until two loads run the same lane in the same week*. Then the signals
   collide and the rulebook guesses.

## The modern approach: retrieve-then-rerank with a fine-tuned transformer

We follow **Ditto** (Li et al., *Deep Entity Matching with Pre-Trained Language
Models*, VLDB 2020). Each document is serialized into a tagged string:

```
[COL] load [VAL] LD100003 [COL] broker [VAL] TQL
[COL] pickup [VAL] Atlanta GA 30301 2025-10-18
[COL] delivery [VAL] Dallas TX 75201 2025-10-19
[COL] weight [VAL] 32000 [COL] rate [VAL] 1100
```

Two models then work together:

1. **Bi-encoder blocking** (`all-MiniLM-L6-v2`). Embed every document once; for each
   BOL retrieve only its **top-k** most similar Rate Cons. This cuts the comparison
   space from *O(BOLs × Rate Cons)* to *O(BOLs × k)* — the efficiency win at scale.
2. **Fine-tuned cross-encoder** (`distilbert-base-uncased`). For each surviving
   candidate, the transformer reads the BOL and Rate Con *jointly* and outputs a match
   probability. Because it sees both records together, it reasons about ambiguous
   cases the rulebook can't.

We train the cross-encoder on labeled pairs from a synthetic generator that injects
realistic extraction noise (OCR character swaps, dropped fields, ±1-day date jitter,
weight rounding, city-name variants) and **hard negatives** (different loads on the
same lane/broker). Training runs on a single RTX 5050 GPU in minutes.

## The benchmark

We evaluate end-to-end matching (predicted BOL↔Rate-Con pairs vs. ground truth) on
400 loads, reporting precision, recall, F1, the number of pairwise comparisons, and
wall-clock time. Two scenarios:

- **Easy** — every load runs a distinct lane (little ambiguity).
- **Hard** — loads are drawn from **8 recurring lanes** within a 21-day window with
  heavy load-number noise. This is the realistic case: a carrier running the same
  lanes repeatedly.

Reproduce with:

```bash
python scripts/benchmark_matching.py --count 400 --noise 0.5            # easy
python scripts/benchmark_matching.py --count 400 --noise 0.7 --lanes 8 --date-window 21   # hard
```

The cross-encoder reaches **F1 1.000 on the held-out pair-classification test set** —
the classifier itself is essentially perfect. End-to-end matching numbers (400 loads,
160,000 possible pairs) below; the ML row uses hybrid blocking (structural ZIP/PO keys
∪ bi-encoder top-10).

**Easy scenario** (distinct lanes, noise 0.5)

| strategy        | precision | recall |  F1   | comparisons |
|-----------------|-----------|--------|-------|-------------|
| heuristic       | 0.964     | 0.950  | 0.957 | 80,937      |
| Fellegi–Sunter  | 1.000     | 0.983  | 0.991 | 81,826      |
| fine-tuned CE   | 1.000     | 1.000  | **1.000** | **6,955** |

**Hard scenario** (8 recurring lanes, 21-day window, noise 0.7 — the realistic case)

| strategy        | precision | recall |  F1   | comparisons |
|-----------------|-----------|--------|-------|-------------|
| heuristic       | 0.667     | 0.640  | 0.653 | 80,630      |
| Fellegi–Sunter  | 0.990     | 0.990  | **0.990** | 80,200  |
| fine-tuned CE   | 0.880     | 0.863  | 0.871 | **15,835**  |

### What the numbers say

- On clean data the heuristic is already strong (F1 0.96); the fine-tuned model is
  **perfect (1.000) using ~12× fewer comparisons** thanks to blocking.
- On **realistic recurring-lane data the heuristic collapses to F1 0.65** — it cannot
  disambiguate near-identical loads once the load number is corrupted.
- **Fellegi–Sunter wins the hard case (F1 0.99).** Learning the field weights (m/u
  probabilities) from labeled data, *with exhaustive comparison*, beats everything when
  the discriminating signal is in a handful of structured fields.
- The fine-tuned cross-encoder recovers the heuristic's collapse (0.65 → **0.87**) at
  **~5× fewer comparisons**, but does not beat exhaustive Fellegi–Sunter here — because
  its accuracy is gated by **blocking recall**, not the classifier (which scores 1.000
  on test pairs).

### The blocking diagnosis (the interesting part)

The cross-encoder's first hard-scenario result was a dismal F1 0.36. The classifier was
fine — the **bi-encoder blocker was the bottleneck**: with ~50 near-identical same-lane
loads, semantic top-5 retrieval simply didn't contain the true Rate Con (recall 0.55;
top-20 only reached 0.85). Pure semantic similarity is the wrong blocking key for
near-duplicate records.

Switching to **hybrid blocking** — union the bi-encoder top-k with *structural* keys
(shared pickup/delivery ZIP, broker PO) — guarantees same-lane candidates are retrieved
and lifts hard-scenario F1 from 0.36 → 0.87 while still cutting comparisons ~5×. The
remaining gap to Fellegi–Sunter is recall lost when *every* structural field is
simultaneously corrupted; closing it means a finer blocking key (e.g. lane × date
bucket) or fine-tuning the bi-encoder.

## Efficiency: blocking

Exhaustive matching of N BOLs against N Rate Cons is N² comparisons (160,000 at N=400).
Hybrid blocking scored each BOL against only its candidate set — **6,955 comparisons on
the easy corpus (~23×) and 15,835 on the hard corpus (~5×)** — and the benchmark prints
the count per strategy so the reduction is measurable, not asserted.

## Takeaways

1. Hand-tuned scoring rules look fine on demo data and quietly fail on the ambiguous
   cases that dominate production (F1 0.96 → 0.65).
2. Even a *classic* learned model (Fellegi–Sunter, 1969) beats hand-tuning — learn your
   weights — and remains a strong, cheap, exhaustive baseline.
3. A fine-tuned transformer is only as good as its **blocking**: a near-perfect
   classifier scored F1 0.36 end-to-end until the candidate generator was fixed.
   Diagnosing *which stage* fails matters more than the model.

## References

- Fellegi, I. P., & Sunter, A. B. (1969). *A Theory for Record Linkage.* JASA.
- Li, Y., Li, J., Suhara, Y., Doan, A., & Tan, W. (2020). *Deep Entity Matching with
  Pre-Trained Language Models.* VLDB.
