from matcher.brokers import detect_broker


def test_tql():
    assert detect_broker("TQL RATE CONFIRMATION FOR PO# 12345") == "TQL"


def test_tql_full_name():
    assert detect_broker("Total Quality Logistics carrier agreement") == "TQL"


def test_ch_robinson():
    assert detect_broker("C.H. Robinson Worldwide load tender") == "CH_ROBINSON"


def test_coyote():
    assert detect_broker("Coyote Logistics broker carrier agreement") == "COYOTE"


def test_echo():
    assert detect_broker("Echo Global Logistics load confirmation") == "ECHO"


def test_landstar():
    assert detect_broker("Landstar System Inc. carrier rate") == "LANDSTAR"


def test_no_match():
    assert detect_broker("random text with no broker name") is None


def test_case_insensitive():
    assert detect_broker("total quality logistics rate con") == "TQL"


def test_only_first_2000_chars():
    # Broker name beyond the 2000-char window should not match
    prefix = "x" * 2001
    assert detect_broker(prefix + "TQL rate confirmation") is None


def test_empty_string():
    assert detect_broker("") is None
