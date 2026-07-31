from extraction.chunker import chunk_text, detect_clause_ref, split_into_paragraphs

COMMON_KWARGS = dict(
    doc_id="doc123",
    section_path="Traffic Commercial Directorate > Commercial Circulars > 2019",
    title="Commercial Circular No. 45 of 2019",
    date="2019-06-12",
    source_url="https://example.com/circular45.pdf",
    local_path="documents/circular45.pdf",
    doc_type="pdf",
)


def _make_paragraph(word_count: int, marker: str = "word") -> str:
    return " ".join(f"{marker}{i}" for i in range(word_count))


def test_split_into_paragraphs_respects_blank_lines():
    text = "Para one.\n\nPara two.\n\nPara three."
    paras = split_into_paragraphs(text)
    assert paras == ["Para one.", "Para two.", "Para three."]


def test_chunks_respect_word_count_bounds():
    # Build enough paragraphs to require multiple chunks in the 200-500 range.
    paragraphs = [_make_paragraph(80) for _ in range(10)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text=text, **COMMON_KWARGS)

    assert len(chunks) >= 2
    for c in chunks[:-1]:  # last chunk may be a smaller remainder
        word_count = len(c.text.split())
        assert word_count <= 500


def test_metadata_propagates_to_every_chunk():
    paragraphs = [_make_paragraph(80) for _ in range(6)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text=text, **COMMON_KWARGS)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == "doc123"
        assert c.section_path == COMMON_KWARGS["section_path"]
        assert c.title == COMMON_KWARGS["title"]
        assert c.date == "2019-06-12"
        assert c.source_url == COMMON_KWARGS["source_url"]
        assert c.local_path == COMMON_KWARGS["local_path"]
        assert c.chunk_id.startswith("doc123::chunk")


def test_chunk_index_is_sequential():
    paragraphs = [_make_paragraph(80) for _ in range(8)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text=text, **COMMON_KWARGS)
    indexes = [c.chunk_index for c in chunks]
    assert indexes == list(range(len(chunks)))


def test_oversized_paragraph_is_split_on_sentence_boundaries():
    # A single paragraph far bigger than max_words, with clause structure.
    sentences = [f"This is sentence number {i} of the clause." for i in range(150)]
    huge_paragraph = " ".join(sentences)

    chunks = chunk_text(text=huge_paragraph, **COMMON_KWARGS)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text.split()) <= 550  # some slack for sentence-boundary splitting


def test_empty_text_produces_no_chunks():
    chunks = chunk_text(text="   \n\n  ", **COMMON_KWARGS)
    assert chunks == []


def test_small_document_produces_single_chunk():
    text = _make_paragraph(50)
    chunks = chunk_text(text=text, **COMMON_KWARGS)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_detect_clause_ref_common_formats():
    assert detect_clause_ref("1. The following shall apply.") == "1"
    assert detect_clause_ref("3.2 Refund procedure is as follows.") == "3.2"
    assert detect_clause_ref("(a) subject to conditions.") == "(a)"
    assert detect_clause_ref("(iii) further provided that.") == "(iii)"
    assert detect_clause_ref("Clause 5.1 states the rate.") == "5.1"
    assert detect_clause_ref("This is a normal sentence with no number.") is None


def test_chunk_text_tags_page_and_clause_from_page_list():
    pages = [
        "1. Introductory clause on page one.\n\n2. Second clause on page one.",
        "2. (continued) Second clause continues onto page two.\n\n3. Third clause starts on page two.",
    ]
    chunks = chunk_text(text=pages, **COMMON_KWARGS)

    assert len(chunks) == 1  # short enough to stay one chunk
    c = chunks[0]
    assert c.page_start == 1
    assert c.page_end == 2
    assert c.clause_ref == "1"  # first paragraph's own clause number


def test_chunk_text_plain_string_treated_as_single_page():
    chunks = chunk_text(text="1. Some clause text here.", **COMMON_KWARGS)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].clause_ref == "1"
