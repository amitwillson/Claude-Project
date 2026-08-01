import types

import anthropic

from qa.answer import answer_question, _format_context
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


class _FakeMessages:
    def __init__(self, capture, reply_text):
        self._capture = capture
        self._reply_text = reply_text

    def create(self, **kwargs):
        self._capture.append(kwargs)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=self._reply_text)])


class _FakeAnthropicClient:
    def __init__(self, capture, reply_text, *a, **k):
        self.messages = _FakeMessages(capture, reply_text)


def test_first_turn_sends_no_prior_history(monkeypatch):
    capture = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: _FakeAnthropicClient(capture, "First answer."))

    answer = answer_question("What is the refund policy?", [_chunk()])

    assert len(capture[0]["messages"]) == 1  # only this turn, no history yet
    assert answer.text == "First answer."
    assert [e["role"] for e in answer.history_entries] == ["user", "assistant"]


def test_follow_up_includes_prior_turns_in_the_request(monkeypatch):
    capture = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: _FakeAnthropicClient(capture, "Second answer."))

    prior_history = [
        {"role": "user", "content": [{"type": "text", "text": "Retrieved excerpts:\n\n...\n\nQuestion: first?"}]},
        {"role": "assistant", "content": "First answer."},
    ]
    answer = answer_question("And clause 5?", [_chunk()], conversation_history=prior_history)

    sent_messages = capture[0]["messages"]
    assert len(sent_messages) == 3  # 2 prior turns + this new user turn
    assert sent_messages[0] == prior_history[0]
    assert sent_messages[1] == prior_history[1]
    assert sent_messages[2]["role"] == "user"
    assert answer.history_entries[1]["content"] == "Second answer."


def test_user_turn_carries_cache_control_for_multi_turn_caching(monkeypatch):
    capture = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: _FakeAnthropicClient(capture, "Answer."))

    answer = answer_question("What is the refund policy?", [_chunk()])

    user_turn = answer.history_entries[0]
    assert user_turn["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_no_chunks_still_returns_history_entries_without_calling_the_api(monkeypatch):
    capture = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: _FakeAnthropicClient(capture, "unused"))

    answer = answer_question("Some obscure question", [])

    assert capture == []  # messages.create never called -- no chunks means no API call
    assert answer.text == "No matching circular found in the indexed documents for this query."
    assert [e["role"] for e in answer.history_entries] == ["user", "assistant"]
