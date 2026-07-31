# Paragraph/clause-aware chunker. Targets ~200-500 words per chunk without
# cutting mid-clause where avoidable. Every chunk carries full source
# metadata -- including the clause/paragraph number and PDF page range it
# came from, where detectable -- so it remains traceable back to its exact
# circular + date + clause even if retrieved in isolation.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

TARGET_MIN_WORDS = 200
TARGET_MAX_WORDS = 500

# Matches a leading clause/paragraph number at the start of a paragraph:
# "1.", "2.3", "10.4.1", "(a)", "(iii)", "Para 5", "Clause 3.2", etc.
# Anchored to the start of the paragraph to avoid matching numbers mid-sentence.
CLAUSE_REF_RE = re.compile(
    r"^\s*(?:(?:Para(?:graph)?|Clause|Cl)\.?\s*)?"
    r"(\d+(?:\.\d+)*\.?|\([a-zA-Z0-9ivxIVX]{1,5}\)|[A-Z]\.)\s+"
)


def detect_clause_ref(paragraph: str) -> Optional[str]:
    """Best-effort extraction of a leading clause/paragraph number, e.g.
    "3.2" from "3.2 The following procedure shall apply...". Returns None
    if the paragraph doesn't appear to start with one -- most continuation
    text won't, and callers should carry forward the last detected ref."""
    m = CLAUSE_REF_RE.match(paragraph)
    if not m:
        return None
    return m.group(1).rstrip(".")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section_path: str
    title: str
    date: Optional[str]
    source_url: str
    local_path: Optional[str]
    chunk_index: int
    doc_type: str = ""
    clause_ref: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: dict = field(default_factory=dict)


def split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines first; if a resulting block is still huge (a wall
    of text with no blank-line breaks), fall back to sentence-ish splitting."""
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks = [b.strip() for b in raw_blocks if b.strip()]

    paragraphs: list[str] = []
    for block in blocks:
        if len(block.split()) <= TARGET_MAX_WORDS * 2:
            paragraphs.append(block)
        else:
            # Very long block with no paragraph breaks: split on sentence
            # boundaries so we still respect clause structure.
            sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z0-9(])", block)
            paragraphs.extend(s.strip() for s in sentences if s.strip())
    return paragraphs


@dataclass
class _TaggedParagraph:
    text: str
    page_number: int
    clause_ref: Optional[str]  # the clause active for this paragraph (own or carried forward)


def _tag_paragraphs(pages: list[str]) -> list[_TaggedParagraph]:
    """Split each page into paragraphs and tag each with its page number and
    the clause/paragraph number in effect (its own leading number, or the
    last one seen -- clauses commonly continue across page breaks)."""
    tagged: list[_TaggedParagraph] = []
    current_clause: Optional[str] = None
    for page_number, page_text in enumerate(pages, start=1):
        for para in split_into_paragraphs(page_text):
            own_ref = detect_clause_ref(para)
            if own_ref:
                current_clause = own_ref
            tagged.append(_TaggedParagraph(text=para, page_number=page_number, clause_ref=current_clause))
    return tagged


def chunk_text(
    text: str | list[str],
    doc_id: str,
    section_path: str,
    title: str,
    date: Optional[str],
    source_url: str,
    local_path: Optional[str],
    doc_type: str = "",
    min_words: int = TARGET_MIN_WORDS,
    max_words: int = TARGET_MAX_WORDS,
) -> list[Chunk]:
    """Greedily pack paragraphs into chunks of roughly min_words-max_words,
    never splitting a paragraph unless it alone exceeds max_words (in which
    case it's split on sentence boundaries so clauses stay intact).

    `text` may be a single string (treated as one page) or a list of
    per-page strings (enables clause_ref/page_start/page_end tracking).
    """
    pages = [text] if isinstance(text, str) else list(text)
    tagged_paragraphs = _tag_paragraphs(pages)
    if not tagged_paragraphs:
        return []

    chunks: list[Chunk] = []
    buffer: list[_TaggedParagraph] = []
    current_words = 0
    index = 0

    def flush():
        nonlocal buffer, current_words, index
        if not buffer:
            return
        chunk_text_value = "\n\n".join(p.text for p in buffer)
        pages_in_chunk = [p.page_number for p in buffer]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk{index}",
                doc_id=doc_id,
                text=chunk_text_value,
                section_path=section_path,
                title=title,
                date=date,
                source_url=source_url,
                local_path=local_path,
                chunk_index=index,
                doc_type=doc_type,
                clause_ref=buffer[0].clause_ref,
                page_start=min(pages_in_chunk),
                page_end=max(pages_in_chunk),
            )
        )
        index += 1
        buffer = []
        current_words = 0

    for tp in tagged_paragraphs:
        para_words = len(tp.text.split())

        if para_words > max_words:
            # Oversized single paragraph: flush what we have, then split
            # this paragraph on sentence boundaries into its own chunk(s).
            flush()
            sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z0-9(])", tp.text)
            sub_parts: list[str] = []
            sub_words = 0

            def flush_sub():
                nonlocal sub_parts, sub_words, index
                if not sub_parts:
                    return
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::chunk{index}",
                        doc_id=doc_id,
                        text=" ".join(sub_parts),
                        section_path=section_path,
                        title=title,
                        date=date,
                        source_url=source_url,
                        local_path=local_path,
                        chunk_index=index,
                        doc_type=doc_type,
                        clause_ref=tp.clause_ref,
                        page_start=tp.page_number,
                        page_end=tp.page_number,
                    )
                )
                index += 1
                sub_parts = []
                sub_words = 0

            for sent in sentences:
                sw = len(sent.split())
                if sub_words + sw > max_words and sub_parts:
                    flush_sub()
                sub_parts.append(sent)
                sub_words += sw
            flush_sub()
            continue

        if current_words + para_words > max_words and current_words >= min_words:
            flush()

        buffer.append(tp)
        current_words += para_words

        if current_words >= min_words and current_words >= max_words:
            flush()

    flush()
    return chunks
