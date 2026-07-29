# Paragraph/clause-aware chunker. Targets ~200-500 words per chunk without
# cutting mid-clause where avoidable. Every chunk carries full source
# metadata so it remains traceable back to its exact circular + date even if
# retrieved in isolation.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

TARGET_MIN_WORDS = 200
TARGET_MAX_WORDS = 500


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


def chunk_text(
    text: str,
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
    case it's split on sentence boundaries so clauses stay intact)."""
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_words = 0
    index = 0

    def flush():
        nonlocal current_parts, current_words, index
        if not current_parts:
            return
        chunk_text_value = "\n\n".join(current_parts)
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
            )
        )
        index += 1
        current_parts = []
        current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words > max_words:
            # Oversized single paragraph: flush what we have, then split
            # this paragraph on sentence boundaries into its own chunk(s).
            flush()
            sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z0-9(])", para)
            sub_parts: list[str] = []
            sub_words = 0
            for sent in sentences:
                sw = len(sent.split())
                if sub_words + sw > max_words and sub_parts:
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
                        )
                    )
                    index += 1
                    sub_parts = []
                    sub_words = 0
                sub_parts.append(sent)
                sub_words += sw
            if sub_parts:
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
                    )
                )
                index += 1
            continue

        if current_words + para_words > max_words and current_words >= min_words:
            flush()

        current_parts.append(para)
        current_words += para_words

        if current_words >= min_words and current_words >= max_words:
            flush()

    flush()
    return chunks
