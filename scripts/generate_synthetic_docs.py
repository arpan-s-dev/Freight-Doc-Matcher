"""Generate synthetic BOLs and Rate Confirmations for demo/testing.

Usage:
    python scripts/generate_synthetic_docs.py                # 26-PDF demo set
    python scripts/generate_synthetic_docs.py --count 200    # 200 matched pairs

Default outputs 26 PDFs to samples/input/:
  - 10 matched BOL+RC pairs across 3 brokers
  - 2 unmatched BOLs
  - 2 unmatched Rate Cons
  - 1 fuzzy-match pair (typo'd load number)

With ``--count N`` it instead renders N procedurally-generated matched pairs
(via ``matcher.linkage.synth.generate_loads``) for a larger demo corpus.
"""

import argparse
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path(__file__).parent.parent / "samples" / "input"

BROKERS = [
    ("TQL", "Total Quality Logistics"),
    ("CH_ROBINSON", "C.H. Robinson"),
    ("COYOTE", "Coyote Logistics"),
]

# Full names for the broader broker set used by the procedural generator.
BROKER_NAMES = {
    "TQL": "Total Quality Logistics", "CH_ROBINSON": "C.H. Robinson",
    "COYOTE": "Coyote Logistics", "LANDSTAR": "Landstar System",
    "ECHO": "Echo Global Logistics", "RXO": "RXO Inc",
}

LOADS = [
    {"load": "LD100001", "po": "34100001", "from_city": "Chicago", "from_state": "IL",
     "from_zip": "60601", "to_city": "Dallas", "to_state": "TX", "to_zip": "75201",
     "pickup": "10/15/2025", "delivery": "10/17/2025", "weight": 24500, "rate": 1850.00},
    {"load": "LD100002", "po": "34100002", "from_city": "Los Angeles", "from_state": "CA",
     "from_zip": "90001", "to_city": "Phoenix", "to_state": "AZ", "to_zip": "85001",
     "pickup": "10/16/2025", "delivery": "10/17/2025", "weight": 18000, "rate": 950.00},
    {"load": "LD100003", "po": "34100003", "from_city": "Atlanta", "from_state": "GA",
     "from_zip": "30301", "to_city": "Nashville", "to_state": "TN", "to_zip": "37201",
     "pickup": "10/18/2025", "delivery": "10/19/2025", "weight": 32000, "rate": 1100.00},
    {"load": "LD100004", "po": "34100004", "from_city": "Seattle", "from_state": "WA",
     "from_zip": "98101", "to_city": "Portland", "to_state": "OR", "to_zip": "97201",
     "pickup": "10/20/2025", "delivery": "10/20/2025", "weight": 15000, "rate": 650.00},
    {"load": "LD100005", "po": "34100005", "from_city": "Houston", "from_state": "TX",
     "from_zip": "77001", "to_city": "San Antonio", "to_state": "TX", "to_zip": "78201",
     "pickup": "10/21/2025", "delivery": "10/22/2025", "weight": 28000, "rate": 780.00},
    {"load": "LD100006", "po": "34100006", "from_city": "Denver", "from_state": "CO",
     "from_zip": "80201", "to_city": "Salt Lake City", "to_state": "UT", "to_zip": "84101",
     "pickup": "10/23/2025", "delivery": "10/24/2025", "weight": 21000, "rate": 1200.00},
    {"load": "LD100007", "po": "34100007", "from_city": "Miami", "from_state": "FL",
     "from_zip": "33101", "to_city": "Orlando", "to_state": "FL", "to_zip": "32801",
     "pickup": "10/25/2025", "delivery": "10/26/2025", "weight": 19000, "rate": 580.00},
    {"load": "LD100008", "po": "34100008", "from_city": "Detroit", "from_state": "MI",
     "from_zip": "48201", "to_city": "Cleveland", "to_state": "OH", "to_zip": "44101",
     "pickup": "10/27/2025", "delivery": "10/28/2025", "weight": 36000, "rate": 720.00},
    {"load": "LD100009", "po": "34100009", "from_city": "Minneapolis", "from_state": "MN",
     "from_zip": "55401", "to_city": "Milwaukee", "to_state": "WI", "to_zip": "53201",
     "pickup": "10/29/2025", "delivery": "10/30/2025", "weight": 22000, "rate": 840.00},
    {"load": "LD100010", "po": "34100010", "from_city": "Kansas City", "from_state": "MO",
     "from_zip": "64101", "to_city": "St. Louis", "to_state": "MO", "to_zip": "63101",
     "pickup": "11/01/2025", "delivery": "11/02/2025", "weight": 29000, "rate": 560.00},
]

CARRIERS = [
    ("Sehajnam Inc", "1553561"),
    ("Swift Transport LLC", "2341234"),
    ("Heartland Express", "3421567"),
]


def _rc(c: canvas.Canvas, load: dict, broker: tuple, carrier_name: str, mc: str) -> None:
    broker_key, broker_full = broker
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 740, f"{broker_full.upper()} RATE CONFIRMATION FOR PO# {load['po']}")
    c.setFont("Helvetica", 10)
    c.drawString(72, 720, f"Load #: {load['load']}")
    c.drawString(72, 700, "CARRIER CONTACT INFO")
    c.drawString(72, 685, f"MC#/DOT# Name Phone Terms")
    c.drawString(72, 670, f"{mc} / 9999999  {carrier_name}  555-000-0000  28DAYS")
    c.drawString(72, 645, "LOAD INFORMATION")
    c.drawString(72, 630, f"${ load['rate']:,.2f} Line Haul Flat  1.0000  ${load['rate']:,.2f}")
    c.drawString(72, 605, "Pick-up Location          Date          Time")
    c.drawString(72, 590, f"{load['from_city']}, {load['from_state']}          {load['pickup']}          08:00")
    c.drawString(72, 570, "Delivery Location          Date          Time")
    c.drawString(72, 555, f"{load['to_city']}, {load['to_state']}          {load['delivery']}          14:00")
    c.drawString(72, 535, f"Estimated Weight {load['weight']}")
    c.drawString(72, 515, f"Origin ZIP: {load['from_zip']}    Destination ZIP: {load['to_zip']}")
    c.showPage()


def _bol(c: canvas.Canvas, load: dict, broker: tuple, load_num: str) -> None:
    broker_key, broker_full = broker
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 740, "BILL OF LADING — STRAIGHT BILL OF LADING")
    c.setFont("Helvetica", 10)
    c.drawString(72, 720, f"BOL Number: {load_num}")
    c.drawString(72, 700, f"Shipper Reference / Load #: {load['load']}")
    c.drawString(72, 680, f"Broker: {broker_full}    PO#: {load['po']}")
    c.drawString(72, 655, "ORIGIN (Ship From)")
    c.drawString(72, 640, f"{load['from_city']}, {load['from_state']} {load['from_zip']}")
    c.drawString(72, 625, f"Pickup Date: {load['pickup']}")
    c.drawString(72, 600, "DESTINATION (Ship To)")
    c.drawString(72, 585, f"{load['to_city']}, {load['to_state']} {load['to_zip']}")
    c.drawString(72, 565, f"Delivery Date: {load['delivery']}")
    c.drawString(72, 545, f"Total Weight: {load['weight']:,} lbs")
    c.showPage()


def _render_pairs(loads: list[dict], brokers, carriers) -> int:
    count = 0
    for i, load in enumerate(loads):
        broker = brokers(i, load)
        carrier_name, mc = carriers(i, load)

        rc_path = OUTPUT_DIR / f"RATECON_{load['load']}_{broker[0]}.pdf"
        c = canvas.Canvas(str(rc_path), pagesize=letter)
        _rc(c, load, broker, carrier_name, mc)
        c.save()

        bol_path = OUTPUT_DIR / f"BOL_{load['load']}.pdf"
        c = canvas.Canvas(str(bol_path), pagesize=letter)
        _bol(c, load, broker, load["load"])
        c.save()

        count += 2
        if (i + 1) % 25 == 0 or i + 1 == len(loads):
            print(f"  Rendered {i + 1}/{len(loads)} pairs")
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=0,
                    help="Render N procedurally-generated matched pairs instead of the demo set.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.count:
        from matcher.linkage.synth import generate_loads
        loads = generate_loads(args.count, args.seed)
        count = _render_pairs(
            loads,
            brokers=lambda i, ld: (ld["broker"], BROKER_NAMES.get(ld["broker"], ld["broker"])),
            carriers=lambda i, ld: ld["carrier"],
        )
        print(f"\nDone — {count} PDFs ({args.count} matched pairs) written to {OUTPUT_DIR}")
        return

    count = 0

    # 10 matched pairs
    for i, load in enumerate(LOADS):
        broker = BROKERS[i % len(BROKERS)]
        carrier_name, mc = CARRIERS[i % len(CARRIERS)]

        rc_path = OUTPUT_DIR / f"RATECON_{load['load']}_{broker[0]}.pdf"
        c = canvas.Canvas(str(rc_path), pagesize=letter)
        _rc(c, load, broker, carrier_name, mc)
        c.save()

        bol_path = OUTPUT_DIR / f"BOL_{load['load']}.pdf"
        c = canvas.Canvas(str(bol_path), pagesize=letter)
        _bol(c, load, broker, load["load"])
        c.save()

        count += 2
        print(f"  Pair {i+1}/10: {load['load']}")

    # 2 unmatched BOLs
    for uid in ["LD200001", "LD200002"]:
        stub = {**LOADS[0], "load": uid, "po": f"ORPHAN{uid[-3:]}"}
        p = OUTPUT_DIR / f"BOL_{uid}_UNMATCHED.pdf"
        c = canvas.Canvas(str(p), pagesize=letter)
        _bol(c, stub, BROKERS[0], uid)
        c.save()
        count += 1
        print(f"  Unmatched BOL: {uid}")

    # 2 unmatched Rate Cons
    for uid in ["LD300001", "LD300002"]:
        stub = {**LOADS[1], "load": uid, "po": f"ORPHANRC{uid[-3:]}"}
        p = OUTPUT_DIR / f"RATECON_{uid}_UNMATCHED.pdf"
        c = canvas.Canvas(str(p), pagesize=letter)
        _rc(c, stub, BROKERS[1], "Orphan Carrier LLC", "8888888")
        c.save()
        count += 1
        print(f"  Unmatched RC:  {uid}")

    # Fuzzy-match pair — BOL has correct ID, RC has typo
    fuzzy_load = {**LOADS[2]}
    bol_path = OUTPUT_DIR / "BOL_LD100003_fuzzy.pdf"
    c = canvas.Canvas(str(bol_path), pagesize=letter)
    _bol(c, fuzzy_load, BROKERS[2], "LD100003")
    c.save()

    fuzzy_rc = {**LOADS[2], "load": "LD1O0003"}  # letter O instead of zero
    rc_path = OUTPUT_DIR / "RATECON_LD100003_fuzzy.pdf"
    c = canvas.Canvas(str(rc_path), pagesize=letter)
    _rc(c, fuzzy_rc, BROKERS[2], "Fuzzy Carrier LLC", "1234567")
    c.save()
    count += 2
    print("  Fuzzy-match pair: LD100003 vs LD1O0003")

    print(f"\nDone — {count} PDFs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
