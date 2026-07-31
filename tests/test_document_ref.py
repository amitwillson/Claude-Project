from extraction.document_ref import detect_document_ref


def test_detects_number_and_dated_line():
    text = (
        "GOVERNMENT OF INDIA\nMINISTRY OF RAILWAYS\n(RAILWAY BOARD)\n\n"
        "No. TC-I/2020/109/1\n\nNew Delhi, dated 23.03.2020\n\n"
        "General Managers, All Zonal Railways\n\nSubject: Refund of unused tickets."
    )
    ref = detect_document_ref(text)
    assert ref.number == "TC-I/2020/109/1"
    assert ref.date_raw == "23.03.2020"
    assert ref.date_parsed == "2020-03-23"


def test_detects_f_no_variant():
    text = "F.No. TCR/1078/2019/2-Part(1)\n\nDated: 12th June 2019"
    ref = detect_document_ref(text)
    assert ref.number == "TCR/1078/2019/2-Part(1)"
    assert ref.date_parsed == "2019-06-12"


def test_ignores_plain_no_of_without_digits():
    text = "No. of copies required shall not exceed thirty per office."
    ref = detect_document_ref(text)
    assert ref.number is None


def test_no_match_returns_none_fields():
    ref = detect_document_ref("This document has no recognizable header at all.")
    assert ref.number is None
    assert ref.date_raw is None
    assert ref.date_parsed is None


def test_empty_text():
    ref = detect_document_ref("")
    assert ref.number is None
    assert ref.date_raw is None
    assert ref.date_parsed is None


def test_ignores_number_beyond_head_window():
    padding = "word " * 400  # pushes well past _HEAD_CHARS
    text = padding + "No. TC-I/2020/109/1 dated 23.03.2020"
    ref = detect_document_ref(text)
    assert ref.number is None
