from qa.answer import Answer, Citation
from qa.retrieval import RetrievedChunk
from qa.validate_citations import run_case


def _chunk(circular_number="", title="Some Circular"):
    return RetrievedChunk(
        chunk_id="c1", text="text", title=title, section_path="A > B", date="2020-01-01",
        source_url="https://x", local_path="documents/x.pdf", score=1.0, source="vector",
        circular_number=circular_number,
    )


def _citation(circular_number="", title="Some Circular"):
    return Citation(
        title=title, date="2020-01-01", section_path="A > B", source_url="https://x",
        local_path="documents/x.pdf", chunk_id="c1", circular_number=circular_number,
    )


def test_passes_when_expected_number_is_cited(monkeypatch):
    monkeypatch.setattr(
        "qa.validate_citations.retrieve",
        lambda *a, **k: ([_chunk(circular_number="TC-I/2020/109/1")], "specific"),
    )
    monkeypatch.setattr(
        "qa.validate_citations.answer_question",
        lambda *a, **k: Answer(
            text="As per TC-I/2020/109/1...", citations=[_citation(circular_number="TC-I/2020/109/1")],
            mode="specific", model="claude-sonnet-5",
        ),
    )
    result = run_case({"question": "Q?", "expected_number": "TC-I/2020/109/1"}, conn=None, vector_store=None, embedder=None)
    assert result.passed


def test_fails_when_expected_number_not_cited(monkeypatch):
    monkeypatch.setattr(
        "qa.validate_citations.retrieve",
        lambda *a, **k: ([_chunk(circular_number="Other/2019/1")], "specific"),
    )
    monkeypatch.setattr(
        "qa.validate_citations.answer_question",
        lambda *a, **k: Answer(
            text="As per Other/2019/1...", citations=[_citation(circular_number="Other/2019/1")],
            mode="specific", model="claude-sonnet-5",
        ),
    )
    result = run_case({"question": "Q?", "expected_number": "TC-I/2020/109/1"}, conn=None, vector_store=None, embedder=None)
    assert not result.passed
    assert "NOT found" in result.reason


def test_passes_on_keyword_fallback_when_number_unknown(monkeypatch):
    monkeypatch.setattr(
        "qa.validate_citations.retrieve",
        lambda *a, **k: ([_chunk(title="Refund of unused tickets policy")], "specific"),
    )
    monkeypatch.setattr(
        "qa.validate_citations.answer_question",
        lambda *a, **k: Answer(
            text="As per the refund circular...", citations=[_citation(title="Refund of unused tickets policy")],
            mode="specific", model="claude-sonnet-5",
        ),
    )
    result = run_case({"question": "Q?", "expected_keyword": "refund"}, conn=None, vector_store=None, embedder=None)
    assert result.passed


def test_fails_when_no_chunks_retrieved(monkeypatch):
    monkeypatch.setattr("qa.validate_citations.retrieve", lambda *a, **k: ([], "specific"))
    monkeypatch.setattr(
        "qa.validate_citations.answer_question",
        lambda *a, **k: Answer(
            text="No matching circular found in the indexed documents for this query.",
            citations=[], mode="specific", model="claude-sonnet-5",
        ),
    )
    result = run_case({"question": "Q?", "expected_number": "X"}, conn=None, vector_store=None, embedder=None)
    assert not result.passed
    assert "no chunks retrieved" in result.reason


def test_case_with_no_expectations_fails_with_clear_reason(monkeypatch):
    monkeypatch.setattr(
        "qa.validate_citations.retrieve",
        lambda *a, **k: ([_chunk()], "specific"),
    )
    monkeypatch.setattr(
        "qa.validate_citations.answer_question",
        lambda *a, **k: Answer(text="...", citations=[_citation()], mode="specific", model="claude-sonnet-5"),
    )
    result = run_case({"question": "Q?"}, conn=None, vector_store=None, embedder=None)
    assert not result.passed
    assert "nothing to check" in result.reason
