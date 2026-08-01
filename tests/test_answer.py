from qa.answer import _format_context
from qa.retrieval import RetrievedChunk


def _chunk(status="current", status_note=""):
    return RetrievedChunk(
        chunk_id="c1", text="some text", title="Some Circular", section_path="A > B",
        date="2020-01-01", source_url="https://x", local_path="documents/x.pdf",
        score=1.0, source="vector", status=status, status_note=status_note,
    )


def test_context_includes_superseded_status_line():
    context = _format_context([_chunk(status="superseded", status_note="New Circular (No. X) dated 2021-01-01")])
    assert "Status: SUPERSEDED by New Circular (No. X) dated 2021-01-01" in context


def test_context_includes_amended_status_line():
    context = _format_context([_chunk(status="amended", status_note="Amended (Clause 4) by X")])
    assert "Status: AMENDED by Amended (Clause 4) by X" in context


def test_context_includes_ambiguous_status_line():
    context = _format_context([_chunk(status="ambiguous", status_note="Conflicting supersession signals: A; B")])
    assert "Status: AMBIGUOUS -- Conflicting supersession signals: A; B" in context


def test_context_omits_status_line_when_current():
    context = _format_context([_chunk()])
    assert "Status:" not in context
