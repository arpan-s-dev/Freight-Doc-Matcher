from datetime import date

import pytest

from matcher.extract_fields import classify_doc_type, extract_fields_regex
from matcher.models import DocType

TQL_SAMPLE = """\
TQL RATE CONFIRMATION FOR PO# 34033724
FIND YOUR NEXT LOAD BY VISITING CARRIERDASHBOARD.TQL.COM
MC#/DOT# Name Phone Terms Fax
1553561 / 4082650 Sehajnam Inc 209-431-7743 28DAYS
$1,100.00 Line Haul Flat 1.0000 $1,100.00
Total: $1,100.00 USD
Pick-up Location Date Time
Temecula, CA 10/9/2025 Appt 10:00
Delivery Location Date Time
Tracy, CA 10/10/2025 Appt 05:30
Estimated Weight 15000
"""


def test_tql_broker_po():
    r = extract_fields_regex(TQL_SAMPLE, broker="TQL")
    assert r["broker_po"] == "34033724"


def test_tql_rate():
    r = extract_fields_regex(TQL_SAMPLE, broker="TQL")
    assert r["rate_amount"] == 1100.0


def test_tql_weight():
    r = extract_fields_regex(TQL_SAMPLE, broker="TQL")
    assert r["weight_lbs"] == 15000.0


def test_tql_pickup_date():
    r = extract_fields_regex(TQL_SAMPLE, broker="TQL")
    assert r["pickup_date"] == date(2025, 10, 9)


def test_tql_delivery_date():
    r = extract_fields_regex(TQL_SAMPLE, broker="TQL")
    assert r["delivery_date"] == date(2025, 10, 10)


def test_tql_pickup_city():
    r = extract_fields_regex(TQL_SAMPLE, broker="TQL")
    assert r.get("pickup_city") == "Temecula"


def test_tql_delivery_city():
    r = extract_fields_regex(TQL_SAMPLE, broker="TQL")
    assert r.get("delivery_city") == "Tracy"


def test_tql_carrier_mc():
    r = extract_fields_regex(TQL_SAMPLE, broker="TQL")
    assert r.get("carrier_mc") == "1553561"


def test_doc_type_rate_con():
    assert classify_doc_type("TQL RATE CONFIRMATION FOR PO# 123", {}) == DocType.RATE_CON


def test_doc_type_bol():
    assert classify_doc_type("BILL OF LADING - Straight Bill of Lading", {}) == DocType.BOL


def test_doc_type_unknown():
    assert classify_doc_type("some random text without keywords", {}) == DocType.UNKNOWN


def test_doc_type_rate_con_via_rate_amount():
    assert classify_doc_type("some load document", {"rate_amount": 1500.0}) == DocType.RATE_CON


def test_mc_number_generic():
    r = extract_fields_regex("MC# 1553561 carrier info")
    assert r.get("carrier_mc") == "1553561"


def test_weight_lbs():
    r = extract_fields_regex("Total weight: 24500 lbs on this shipment")
    assert r.get("weight_lbs") == 24500.0


def test_zip_codes():
    r = extract_fields_regex("pickup 92590 delivery 95376")
    assert r.get("pickup_zip") == "92590"
    assert r.get("delivery_zip") == "95376"


def test_cid_garbage_ignored():
    r = extract_fields_regex("(cid:2)(cid:3)" * 50)
    # Should not crash; doc_type should be UNKNOWN
    assert r["doc_type"] == DocType.UNKNOWN
