from extraction.pdf_extract import ExtractionResult, flag_extraction_outliers


def test_flags_low_word_count_relative_to_pages():
    # A 10-page circular that extracted to 40 words almost certainly lost content.
    result = ExtractionResult(
        text="word " * 40, page_count=10, word_count=40,
        has_text_layer=True, ocr_used=False,
    )
    assert flag_extraction_outliers(result) is True


def test_does_not_flag_healthy_extraction():
    result = ExtractionResult(
        text="word " * 3000, page_count=10, word_count=3000,
        has_text_layer=True, ocr_used=False,
    )
    assert flag_extraction_outliers(result) is False


def test_flags_extraction_with_error():
    result = ExtractionResult(
        text="", page_count=0, word_count=0,
        has_text_layer=False, ocr_used=False, error="file could not be opened",
    )
    assert flag_extraction_outliers(result) is True


def test_flags_zero_page_count():
    result = ExtractionResult(
        text="", page_count=0, word_count=0,
        has_text_layer=False, ocr_used=False,
    )
    assert flag_extraction_outliers(result) is True


def test_flags_low_confidence_ocr():
    result = ExtractionResult(
        text="word " * 500, page_count=5, word_count=500,
        has_text_layer=False, ocr_used=True, ocr_confidence=30.0,
    )
    assert flag_extraction_outliers(result) is True


def test_does_not_flag_high_confidence_ocr_with_good_word_count():
    result = ExtractionResult(
        text="word " * 500, page_count=5, word_count=500,
        has_text_layer=False, ocr_used=True, ocr_confidence=85.0,
    )
    assert flag_extraction_outliers(result) is False


def test_custom_threshold():
    result = ExtractionResult(
        text="word " * 100, page_count=10, word_count=100,
        has_text_layer=True, ocr_used=False,
    )
    # 10 words/page fails a threshold of 30 but passes a lower threshold of 5.
    assert flag_extraction_outliers(result, words_per_page_threshold=30) is True
    assert flag_extraction_outliers(result, words_per_page_threshold=5) is False
