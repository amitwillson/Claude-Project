# Builds the system prompt and calls the Anthropic API to answer a question
# using only the retrieved chunks — never general background knowledge.
from __future__ import annotations

import os
from dataclasses import dataclass

from typing import Optional

from qa.retrieval import RetrievedChunk

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a Q&A assistant for the Indian Railways Traffic Commercial \
Directorate circular knowledge base. You answer questions ONLY using the retrieved \
circular excerpts provided to you in this conversation. Follow these rules strictly:

1. Answer ONLY from the provided excerpts. Do NOT use general or background knowledge \
about Indian Railways policy to fill gaps, even if you believe you know the answer. \
If the excerpts do not clearly answer the question, say so explicitly: \
"No matching circular found in the indexed documents for this query." Do not guess or \
extrapolate.

2. ALWAYS cite the exact circular number/title and date for every claim, in the form: \
"As per Commercial Circular No. 45 of 2019 dated 12.06.2019...". Each excerpt is \
labeled with a "Notification/Letter No." detected directly from the document itself, \
which is the most authoritative source for the circular's own number -- prefer it over \
a number merely mentioned in the title when both are present. If neither is available, \
cite the title and date as printed instead. Never state a policy without a citation.

3. Every excerpt is labeled with a Clause and Page (e.g. "Clause: 3.2", "Page: 2" or \
"Pages: 2-3"). Whenever the excerpt has a clause number, include it in the citation, \
e.g. "As per Commercial Circular No. 45 of 2019 dated 12.06.2019, Clause 3.2, page 2...". \
If no clause number was detected for that excerpt, cite the page number alone \
(e.g. "page 2"). If neither is available, cite circular/title/date only as in rule 2.

4. Some excerpts carry a Status line -- the knowledge base has already resolved these \
against other indexed circulars, at the level of the specific clause each excerpt is \
from, not just the whole document:
   - "Status: SUPERSEDED by <circular>" -- this exact clause/passage was replaced in \
full. Treat it as an old rule, not the current one: lead your answer with the \
superseding circular's rule, and only mention this one for historical context (e.g. \
"this was the rule under Circular X until it was superseded by Circular Y, which now \
governs..."), never as the primary answer.
   - "Status: AMENDED by <circular>" -- ONLY this specific clause was changed by a later \
circular; the rest of this document (its other clauses/excerpts) remains in force \
unless separately marked otherwise. COMBINE both in your answer: state the current rule \
using the amendment for this clause, and other clauses from this same circular (if also \
retrieved, with no Status line of their own) as still valid -- do not treat the whole \
document as replaced just because one clause was amended.
   - "Status: AMBIGUOUS -- <details>" -- the knowledge base found conflicting \
supersession signals for this exact passage (e.g. one document claims to fully replace \
it while another, differently dated, claims to have only amended it) and deliberately \
did not pick one. Tell the user about the conflict explicitly and recommend verifying \
against the original documents rather than asserting either version as current.
   - No Status line at all means nothing in the index contradicts this excerpt -- treat \
it as current.
   If excerpts conflict WITHOUT any of the above Status labels (i.e. the knowledge base's \
own detection didn't catch it, but the text itself suggests a conflict, or an excerpt's \
own text states it supersedes/amends/is issued "in partial modification of" another \
circular not reflected in a Status line), surface that explicitly instead of silently \
picking one -- tell the user which is more recent (by date) if determinable, and flag \
that supersession detection is heuristic and the user may want to verify against the \
original documents.

5. If asked to list/summarize many circulars, cover every relevant one shown in the \
excerpts you were given -- do not stop at the first few.

6. Be concise and precise. Do not editorialize beyond what the source text supports.

7. Some excerpts are placeholders beginning with "[No machine-readable text could be \
extracted from this document by OCR...]" -- this means a real, indexed circular exists \
and is relevant by title/date/section, but its content could not be automatically read. \
Tell the user such a circular appears relevant and point them to the original PDF; do \
NOT treat the placeholder as if the circular contains no policy, and do NOT say "no \
matching circular found" when a placeholder excerpt was retrieved.

8. This may be a continuing conversation -- earlier questions and answers may appear \
before the current one. You may refer back to them for conversational continuity (e.g. \
"as mentioned above..."), but every NEW factual claim in your answer must still be \
grounded in the excerpts provided WITH THIS question, cited per the rules above -- never \
answer a follow-up purely from memory of an earlier turn without re-grounding it in the \
excerpts you were just given. If this turn's excerpts don't cover the follow-up, say so \
explicitly rather than reusing an earlier answer's citation for a claim it doesn't \
actually support."""


@dataclass
class Citation:
    title: str
    date: str
    section_path: str
    source_url: str
    local_path: str
    chunk_id: str
    clause_ref: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    circular_number: str = ""
    status: str = "current"
    status_note: str = ""


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    mode: str
    model: str
    # The exact API message objects for this turn (user question + assistant
    # reply), in the shape the Anthropic `messages` param expects. Callers
    # doing multi-turn chat should append these to their running history and
    # pass that history back in as `conversation_history` on the next call --
    # see qa/dashboard.py. Single source of truth for the turn's exact
    # content/cache_control placement, so a caller never has to reconstruct
    # it (and risk a byte mismatch that would silently break caching).
    history_entries: Optional[list[dict]] = None


def page_label(page_start: Optional[int], page_end: Optional[int]) -> str:
    if not page_start:
        return "unknown"
    if page_end and page_end != page_start:
        return f"{page_start}-{page_end}"
    return str(page_start)


_STATUS_LABELS = {
    "superseded": "SUPERSEDED by",
    "amended": "AMENDED by",
    "ambiguous": "AMBIGUOUS --",
}


def _status_line(status: str, status_note: str) -> str:
    if status == "current" or not status_note:
        return ""
    label = _STATUS_LABELS.get(status, status.upper())
    return f"Status: {label} {status_note}\n"


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        status_line = _status_line(c.status, c.status_note)
        blocks.append(
            f"[Excerpt {i}]\n"
            f"Title: {c.title}\n"
            f"Date: {c.date or 'unknown'}\n"
            f"Notification/Letter No.: {c.circular_number or 'not detected'}\n"
            f"Section: {c.section_path}\n"
            f"Clause: {c.clause_ref or 'none detected'}\n"
            f"Page: {page_label(c.page_start, c.page_end)}\n"
            f"{status_line}"
            f"Source URL: {c.source_url}\n"
            f"---\n{c.text}\n"
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "No retrieved excerpts were found in the knowledge base for this question."
        )
    context = _format_context(chunks)
    return f"Retrieved excerpts:\n\n{context}\n\nQuestion: {question}"


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    mode: str = "specific",
    model: str | None = None,
    api_key: str | None = None,
    conversation_history: Optional[list[dict]] = None,
) -> Answer:
    """Call the Anthropic API to answer `question` using only `chunks`.

    `conversation_history` (optional): prior turns as a list of Anthropic
    `messages`-shaped dicts, for a continuing chat -- see
    Answer.history_entries. Each turn's excerpts are still grounded fresh
    (rule 8 in SYSTEM_PROMPT); history is for conversational continuity,
    not a substitute for re-retrieving/re-citing on every question."""
    import anthropic  # lazy import

    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    user_message = build_user_message(question, chunks)
    # Stored/replayed in plain form (no cache_control) -- this is what goes
    # into Answer.history_entries and gets threaded back in as next turn's
    # conversation_history. Deliberately NOT cached here: the API caps a
    # request at 4 cache_control breakpoints total, and if every past turn
    # kept the marker it was tagged with when it was the newest turn, a
    # 4-5 question conversation would exceed that cap and start 400ing.
    user_turn = {"role": "user", "content": user_message}

    if not chunks:
        answer_text = "No matching circular found in the indexed documents for this query."
        return Answer(
            text=answer_text,
            citations=[],
            mode=mode,
            model=model,
            history_entries=[user_turn, {"role": "assistant", "content": answer_text}],
        )

    # cache_control goes on the LATEST turn only, for THIS request -- never
    # persisted into history_entries above. Each subsequent call re-applies
    # it fresh to whatever is newest, so the request always carries exactly
    # one breakpoint here (plus one for the system prompt) regardless of how
    # long the conversation grows. Per shared/prompt-caching.md "Multi-turn
    # conversations": a breakpoint on the latest turn is enough for the next
    # call to reuse the whole prior prefix -- earlier turns don't need their
    # own marker for that to work.
    cached_user_turn = {
        "role": "user",
        "content": [{"type": "text", "text": user_message, "cache_control": {"type": "ephemeral"}}],
    }
    messages = list(conversation_history or []) + [cached_user_turn]
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        # SYSTEM_PROMPT is identical on every call -- cache it so repeated
        # questions only pay full price for the (small) per-question user
        # message, not the whole instruction set each time. No effect on
        # behavior; below the model's minimum cacheable prefix it's simply
        # a no-op (no error, cache_creation_input_tokens stays 0).
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )

    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    answer_text = "\n".join(text_parts).strip()

    citations = [
        Citation(
            title=c.title,
            date=c.date,
            section_path=c.section_path,
            source_url=c.source_url,
            local_path=c.local_path,
            chunk_id=c.chunk_id,
            clause_ref=c.clause_ref,
            page_start=c.page_start,
            page_end=c.page_end,
            circular_number=c.circular_number,
            status=c.status,
            status_note=c.status_note,
        )
        for c in chunks
    ]

    return Answer(
        text=answer_text,
        citations=citations,
        mode=mode,
        model=model,
        history_entries=[user_turn, {"role": "assistant", "content": answer_text}],
    )
