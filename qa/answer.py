# Builds the system prompt and calls the Anthropic API to answer a question
# using only the retrieved chunks — never general background knowledge.
from __future__ import annotations

import os
from dataclasses import dataclass

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
"As per Commercial Circular No. 45 of 2019 dated 12.06.2019...". If a circular number \
is not present in the excerpt, cite the title and date as printed instead. Never state \
a policy without a citation.

3. If multiple excerpts appear to conflict, or one excerpt states it supersedes, \
amends, or is issued "in partial modification of" another, surface that explicitly in \
your answer rather than silently picking one as authoritative. Tell the user which \
circular is more recent (by date) if that is determinable, and flag that supersession \
detection is heuristic and the user may want to verify against the original documents.

4. If asked to list/summarize many circulars, cover every relevant one shown in the \
excerpts you were given -- do not stop at the first few.

5. Be concise and precise. Do not editorialize beyond what the source text supports."""


@dataclass
class Citation:
    title: str
    date: str
    section_path: str
    source_url: str
    chunk_id: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    mode: str
    model: str


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Excerpt {i}]\n"
            f"Title: {c.title}\n"
            f"Date: {c.date or 'unknown'}\n"
            f"Section: {c.section_path}\n"
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
) -> Answer:
    """Call the Anthropic API to answer `question` using only `chunks`."""
    import anthropic  # lazy import

    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    if not chunks:
        return Answer(
            text="No matching circular found in the indexed documents for this query.",
            citations=[],
            mode=mode,
            model=model,
        )

    user_message = build_user_message(question, chunks)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    answer_text = "\n".join(text_parts).strip()

    citations = [
        Citation(
            title=c.title,
            date=c.date,
            section_path=c.section_path,
            source_url=c.source_url,
            chunk_id=c.chunk_id,
        )
        for c in chunks
    ]

    return Answer(text=answer_text, citations=citations, mode=mode, model=model)
