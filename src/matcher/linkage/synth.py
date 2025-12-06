"""Procedural synthetic freight data for training and benchmarking the matcher.

Two layers:

* ``generate_loads`` returns plain load dicts (same shape as the hardcoded
  ``LOADS`` in ``scripts/generate_synthetic_docs.py``) so the PDF generator can
  reuse it for a ``--count`` flag.
* ``build_pairs`` turns loads into labeled ``(doc_a, doc_b, label)`` triples of
  ``ExtractedDocument`` records — positives (a BOL and its noised Rate Con) plus
  hard and random negatives — which is exactly what the entity-matching model and
  the benchmark consume. Records represent *post-extraction* output, so we can
  inject realistic OCR/extraction noise without rendering and re-OCRing PDFs.

Everything is seeded (``random.Random(seed)``) for reproducible training runs,
benchmarks, and screenshots.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from matcher.models import DocType, ExtractedDocument, SourceType

# (city, state, zip) pool — real US freight lanes, enough variety for hard negatives.
CITIES = [
    ("Chicago", "IL", "60601"), ("Dallas", "TX", "75201"), ("Los Angeles", "CA", "90001"),
    ("Phoenix", "AZ", "85001"), ("Atlanta", "GA", "30301"), ("Nashville", "TN", "37201"),
    ("Seattle", "WA", "98101"), ("Portland", "OR", "97201"), ("Houston", "TX", "77001"),
    ("San Antonio", "TX", "78201"), ("Denver", "CO", "80201"), ("Salt Lake City", "UT", "84101"),
    ("Miami", "FL", "33101"), ("Orlando", "FL", "32801"), ("Detroit", "MI", "48201"),
    ("Cleveland", "OH", "44101"), ("Minneapolis", "MN", "55401"), ("Milwaukee", "WI", "53201"),
    ("Kansas City", "MO", "64101"), ("St. Louis", "MO", "63101"), ("Memphis", "TN", "38101"),
    ("Charlotte", "NC", "28201"), ("Columbus", "OH", "43201"), ("Indianapolis", "IN", "46201"),
    ("Newark", "NJ", "07101"), ("Boston", "MA", "02101"),
]

BROKERS = ["TQL", "CH_ROBINSON", "COYOTE", "LANDSTAR", "ECHO", "RXO"]

CARRIERS = [
    ("Sehajnam Inc", "1553561"), ("Swift Transport LLC", "2341234"),
    ("Heartland Express", "3421567"), ("Werner Enterprises", "4102938"),
]

# City-name variants an OCR pass or a broker template might emit.
_CITY_VARIANTS = {
    "St. Louis": ["St Louis", "Saint Louis"],
    "Los Angeles": ["L.A.", "Los Angeles "],
    "San Antonio": ["San Antonio ", "S. Antonio"],
    "Salt Lake City": ["SLC", "Salt Lake Cty"],
    "Kansas City": ["Kansas Cty", "KC"],
}


def generate_loads(count: int, seed: int = 42, n_lanes: int | None = None,
                   date_window: int = 120) -> list[dict]:
    """Generate ``count`` load dicts.

    By default each load gets a distinct origin/destination lane (an easy matching
    problem). Pass ``n_lanes`` to draw from a small pool of *recurring* lanes and a
    tighter ``date_window`` — this mimics a carrier running the same lanes daily, so
    many loads share lane/broker/weight/dates and become genuinely ambiguous once
    load numbers are corrupted (the realistic hard case for entity matching).
    """
    rng = random.Random(seed)
    base = date(2025, 10, 1)
    lane_pool = [tuple(rng.sample(CITIES, 2)) for _ in range(n_lanes)] if n_lanes else None
    loads: list[dict] = []
    for i in range(count):
        if lane_pool:
            origin, dest = rng.choice(lane_pool)
        else:
            origin, dest = rng.sample(CITIES, 2)
        pickup = base + timedelta(days=rng.randint(0, date_window))
        transit = rng.randint(1, 4)
        delivery = pickup + timedelta(days=transit)
        loads.append({
            "load": f"LD{100001 + i:06d}",
            "po": f"34{100001 + i:06d}",
            "from_city": origin[0], "from_state": origin[1], "from_zip": origin[2],
            "to_city": dest[0], "to_state": dest[1], "to_zip": dest[2],
            "pickup": pickup.strftime("%m/%d/%Y"),
            "delivery": delivery.strftime("%m/%d/%Y"),
            "weight": rng.randrange(8000, 44000, 500),
            "rate": round(rng.uniform(450, 3200), 2),
            "broker": BROKERS[i % len(BROKERS)],
            "carrier": CARRIERS[i % len(CARRIERS)],
        })
    return loads


def _parse(mdy: str) -> date:
    m, d, y = (int(x) for x in mdy.split("/"))
    return date(y, m, d)


def _bol_from_load(load: dict) -> ExtractedDocument:
    return ExtractedDocument(
        source_path=Path(f"synthetic/BOL_{load['load']}.pdf"),
        doc_type=DocType.BOL,
        source_type=SourceType.NATIVE_PDF,
        broker=load["broker"],
        load_number=load["load"],
        broker_po=load["po"],
        pickup_date=_parse(load["pickup"]),
        pickup_city=load["from_city"], pickup_state=load["from_state"], pickup_zip=load["from_zip"],
        delivery_date=_parse(load["delivery"]),
        delivery_city=load["to_city"], delivery_state=load["to_state"], delivery_zip=load["to_zip"],
        weight_lbs=float(load["weight"]),
        extraction_method="synthetic",
        confidence=1.0,
    )


def _rc_from_load(load: dict, rng: random.Random, noise: float) -> ExtractedDocument:
    """A Rate Con for the same load, with optional extraction noise applied."""
    load_number = load["load"]
    po = load["po"]
    pickup = _parse(load["pickup"])
    delivery = _parse(load["delivery"])
    weight = float(load["weight"])
    pickup_city = load["from_city"]
    delivery_city = load["to_city"]
    pickup_zip = load["from_zip"]

    if noise:
        roll = rng.random()
        if roll < noise * 0.3:     # broker didn't print the load # on the rate con
            load_number = None
        elif roll < noise:         # OCR-style load-number corruption (0 -> O, transpose)
            load_number = _corrupt_load(load_number, rng, heavy=roll < noise * 0.6)
        if rng.random() < noise:  # ±1 day appointment vs. ship date
            delivery = delivery + timedelta(days=rng.choice([-1, 1]))
        if rng.random() < noise:  # weight rounded/estimated on the rate con
            weight = float(round(weight / 1000.0) * 1000)
        if rng.random() < noise:  # broker drops the PO on the rate con
            po = None
        if rng.random() < noise:  # city name variant / abbreviation
            pickup_city = _vary_city(pickup_city, rng)
        if rng.random() < noise * 0.5:  # zip occasionally missing from the rate con
            pickup_zip = None

    carrier_name, mc = load["carrier"]
    return ExtractedDocument(
        source_path=Path(f"synthetic/RATECON_{load['load']}.pdf"),
        doc_type=DocType.RATE_CON,
        source_type=SourceType.NATIVE_PDF,
        broker=load["broker"],
        load_number=load_number,
        broker_po=po,
        pickup_date=pickup,
        pickup_city=pickup_city, pickup_state=load["from_state"], pickup_zip=pickup_zip,
        delivery_date=delivery,
        delivery_city=delivery_city, delivery_state=load["to_state"], delivery_zip=load["to_zip"],
        weight_lbs=weight,
        rate_amount=load["rate"],
        carrier_name=carrier_name, carrier_mc=mc,
        extraction_method="synthetic",
        confidence=1.0,
    )


def _corrupt_load(load_number: str, rng: random.Random, heavy: bool = False) -> str:
    chars = list(load_number)
    swaps = {"0": "O", "1": "I", "5": "S", "8": "B", "O": "0", "I": "1"}
    for _ in range(2 if heavy else 1):
        idx = rng.randrange(len(chars))
        chars[idx] = swaps.get(chars[idx], rng.choice("0123456789"))
    return "".join(chars)


def _vary_city(city: str, rng: random.Random) -> str:
    return rng.choice(_CITY_VARIANTS.get(city, [city]))


@dataclass
class LabeledPair:
    bol: ExtractedDocument
    rc: ExtractedDocument
    label: int  # 1 = match, 0 = non-match


def build_pairs(
    count: int = 500,
    seed: int = 42,
    noise: float = 0.4,
    hard_neg_per_pos: int = 1,
    rand_neg_per_pos: int = 1,
    n_lanes: int | None = None,
) -> list[LabeledPair]:
    """Build labeled BOL/Rate-Con pairs: positives + hard + random negatives.

    Hard negatives share a lane origin or broker with the anchor BOL (so the model
    must rely on more than coarse signals); random negatives are drawn uniformly.
    ``n_lanes`` forwards the recurring-lane (hard) scenario to ``generate_loads``.
    """
    rng = random.Random(seed)
    loads = generate_loads(count, seed, n_lanes=n_lanes)
    bols = [_bol_from_load(ld) for ld in loads]
    rcs = [_rc_from_load(ld, rng, noise) for ld in loads]

    pairs: list[LabeledPair] = []
    for i, (bol, rc) in enumerate(zip(bols, rcs)):
        pairs.append(LabeledPair(bol, rc, 1))

        # Hard negatives: other RCs sharing origin zip or broker, different load.
        hard_pool = [
            j for j in range(len(rcs))
            if j != i and (rcs[j].pickup_zip == bol.pickup_zip or rcs[j].broker == bol.broker)
        ]
        for j in rng.sample(hard_pool, min(hard_neg_per_pos, len(hard_pool))):
            pairs.append(LabeledPair(bol, rcs[j], 0))

        # Random negatives.
        for _ in range(rand_neg_per_pos):
            j = rng.randrange(len(rcs))
            if j != i:
                pairs.append(LabeledPair(bol, rcs[j], 0))

    rng.shuffle(pairs)
    return pairs


def split_pairs(
    pairs: list[LabeledPair], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[list[LabeledPair], list[LabeledPair], list[LabeledPair]]:
    """Deterministic train/val/test split."""
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test
