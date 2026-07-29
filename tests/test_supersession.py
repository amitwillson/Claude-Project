from extraction.supersession import find_supersession_refs


def test_detects_supersession_pattern():
    text = (
        "This circular is issued in supersession of Commercial Circular No. 5 of 2018 "
        "dated 01.01.2018. All Zonal Railways are advised accordingly."
    )
    refs = find_supersession_refs(text)
    assert len(refs) == 1
    ref = refs[0]
    assert ref.relation == "supersession"
    assert ref.referenced_number == "5"
    assert ref.referenced_year == "2018"
    assert ref.referenced_date_raw == "01.01.2018"


def test_detects_partial_modification_pattern():
    text = "In partial modification of Circular No.45/2019 dated 12-06-2019, the following is added."
    refs = find_supersession_refs(text)
    assert len(refs) == 1
    assert refs[0].relation == "partial_modification"
    assert refs[0].referenced_number == "45/2019"


def test_detects_amendment_pattern():
    text = "This circular amends the fare structure notified earlier."
    refs = find_supersession_refs(text)
    assert len(refs) == 1
    assert refs[0].relation == "amendment"


def test_detects_continuation_pattern():
    text = "In continuation of letter no. TC-II/2020/45 dated 3rd March 2020, please note."
    refs = find_supersession_refs(text)
    assert len(refs) == 1
    assert refs[0].relation == "continuation"
    assert refs[0].referenced_date_raw == "3rd March 2020"


def test_no_false_positive_on_plain_text():
    text = "This circular clarifies the refund policy for cancelled tickets."
    refs = find_supersession_refs(text)
    assert refs == []


def test_multiple_supersession_sentences_in_one_document():
    text = (
        "In supersession of Commercial Circular No. 1 of 2015 dated 01.01.2015, this is issued. "
        "Separately, in partial modification of Circular No. 9 of 2017 dated 05.05.2017, rates are revised."
    )
    refs = find_supersession_refs(text)
    relations = {r.relation for r in refs}
    assert relations == {"supersession", "partial_modification"}
