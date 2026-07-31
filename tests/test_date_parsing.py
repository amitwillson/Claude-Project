from scraper.date_parsing import parse_date


def test_dd_dot_mm_dot_yyyy():
    assert parse_date("12.06.2019") == "2019-06-12"


def test_dd_dash_mm_dash_yyyy():
    assert parse_date("15-08-2019") == "2019-08-15"


def test_dd_slash_mm_slash_yyyy():
    assert parse_date("01/01/2018") == "2018-01-01"


def test_dd_mon_yyyy_abbreviated():
    assert parse_date("12 Jun 2019") == "2019-06-12"


def test_dd_mon_yyyy_full_month():
    assert parse_date("12 June 2019") == "2019-06-12"


def test_dd_mon_yyyy_with_ordinal_suffix():
    assert parse_date("3rd March 2021") == "2021-03-03"


def test_dd_mon_yyyy_with_comma():
    assert parse_date("12 Jun, 2019") == "2019-06-12"


def test_date_embedded_in_sentence():
    assert parse_date("Dated: 12.06.2020") == "2020-06-12"
    assert parse_date("This circular dated 3rd March 2021 supersedes...") == "2021-03-03"


def test_two_digit_year_heuristic():
    # 2-digit years are rare on this site but should not crash; heuristic
    # maps <=69 to 2000s, else 1900s.
    assert parse_date("01.01.20") == "2020-01-01"


def test_invalid_date_returns_none():
    assert parse_date("35.13.2019") is None


def test_empty_and_none_input():
    assert parse_date("") is None
    assert parse_date(None) is None


def test_no_date_present():
    assert parse_date("no date here") is None
