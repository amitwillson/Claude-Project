from qa.answer import _format_context
from qa.retrieval import RetrievedChunk


def _chunk(superseded_by_summary=""):
    return RetrievedChunk(
        chunk_id="c1", text="some text", title="Some Circular", section_path="A > B",
        date="2020-01-01", source_url="https://x", local_path="documents/x.pdf",
        score=1.0, source="vector", superseded_by_summary=superseded_by_summary,
    )


def test_context_includes_status_line_when_superseded():
    context = _format_context([_chunk(superseded_by_summary="New Circular (No. X) dated 2021-01-01")])
    assert "Status: SUPERSEDED by New Circular (No. X) dated 2021-01-01" in context


def test_context_omits_status_line_when_not_superseded():
    context = _format_context([_chunk()])
    assert "Status:" not in context
