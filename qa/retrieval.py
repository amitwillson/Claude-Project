# Retrieval: specific mode (vector top-N) + exhaustive mode (large top-N +
# FTS5 keyword union). Exhaustive mode triggers on phrasing like "list all",
# "every circular", "summarize all", "all rules on", or an explicit flag.
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

SPECIFIC_TOP_N = 6
EXHAUSTIVE_TOP_N = 40
EXHAUSTIVE_FTS_LIMIT = 60

_EXHAUSTIVE_TRIGGERS = [
    re.compile(r"\blist\s+all\b", re.IGNORECASE),
    re.compile(r"\bevery\s+circular\b", re.IGNORECASE),
    re.compile(r"\ball\s+circulars?\b", re.IGNORECASE),
    re.compile(r"\bsummarize\s+all\b", re.IGNORECASE),
    re.compile(r"\ball\s+rules?\s+on\b", re.IGNORECASE),
    re.compile(r"\ball\s+of\s+the\b", re.IGNORECASE),
    re.compile(r"\bcomplete\s+list\b", re.IGNORECASE),
]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    section_path: str
    date: str
    source_url: str
    local_path: str
    score: float
    source: str  # "vector" or "keyword"
    clause_ref: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    circular_number: str = ""
    status: str = "current"  # current | superseded | amended | ambiguous
    status_note: str = ""


def is_exhaustive_query(question: str) -> bool:
    return any(pattern.search(question) for pattern in _EXHAUSTIVE_TRIGGERS)


def retrieve(
    question: str,
    conn: sqlite3.Connection,
    vector_store=None,
    embedder=None,
    exhaustive: bool | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """Retrieve relevant chunks. Returns (chunks, mode) where mode is
    "specific" or "exhaustive". FTS5 keyword search always runs (per
    addendum B); exhaustive mode additionally pulls a much larger candidate
    set and unions vector + keyword results so nothing relevant is skipped
    purely for scoring slightly below a cutoff."""
    if exhaustive is None:
        exhaustive = is_exhaustive_query(question)

    top_n = EXHAUSTIVE_TOP_N if exhaustive else SPECIFIC_TOP_N
    fts_limit = EXHAUSTIVE_FTS_LIMIT if exhaustive else SPECIFIC_TOP_N

    results: dict[str, RetrievedChunk] = {}

    if vector_store is not None and embedder is not None:
        try:
            query_embedding = embedder.embed([question])[0]
            hits = vector_store.query(query_embedding, top_n=top_n)
            for hit in hits:
                meta = hit["metadata"] or {}
                # Chroma distance: smaller is better; convert to a similarity-ish score.
                distance = hit.get("distance")
                score = 1.0 / (1.0 + distance) if distance is not None else 0.0
                results[hit["chunk_id"]] = RetrievedChunk(
                    chunk_id=hit["chunk_id"],
                    text=hit["text"],
                    title=meta.get("title", ""),
                    section_path=meta.get("section_path", ""),
                    date=meta.get("date", ""),
                    source_url=meta.get("source_url", ""),
                    local_path=meta.get("local_path", ""),
                    score=score,
                    source="vector",
                    clause_ref=meta.get("clause_ref", "") or "",
                    page_start=meta.get("page_start") or None,
                    page_end=meta.get("page_end") or None,
                    circular_number=meta.get("circular_number", "") or "",
                    status=meta.get("status", "") or "current",
                    status_note=meta.get("status_note", "") or "",
                )
        except Exception:
            pass  # vector store unavailable -> fall through to keyword-only

    from storage.db import fts_search

    for row in fts_search(conn, question, limit=fts_limit):
        cid = row["chunk_id"]
        if cid in results:
            continue  # already have a (higher-fidelity) vector hit
        results[cid] = RetrievedChunk(
            chunk_id=cid,
            text=row["text"],
            title=row["title"],
            section_path=row["section_path"],
            date=row["date"] or "",
            source_url=row["source_url"],
            local_path=row["local_path"] or "",
            score=0.0,
            source="keyword",
            clause_ref=row["clause_ref"] or "" if "clause_ref" in row.keys() else "",
            page_start=row["page_start"] if "page_start" in row.keys() else None,
            page_end=row["page_end"] if "page_end" in row.keys() else None,
            circular_number=row["circular_number"] or "" if "circular_number" in row.keys() else "",
            status=(row["status"] or "current") if "status" in row.keys() else "current",
            status_note=row["status_note"] or "" if "status_note" in row.keys() else "",
        )

    ordered = sorted(results.values(), key=lambda r: r.score, reverse=True)
    if not exhaustive:
        # Prefer non-fully-superseded chunks in the default "specific" mode
        # so an old, replaced rule doesn't crowd a stale answer into the
        # top_n cutoff -- but never drop to zero results just because
        # everything relevant happens to be superseded (e.g. the successor
        # wasn't indexed, or resolution is incomplete): fall back to the
        # unfiltered set rather than hide a possibly-still-useful answer.
        # 'amended' and 'ambiguous' chunks are always kept even here: an
        # 'amended' chunk's clause may be the ONLY current source for that
        # specific clause (the rest of its document is unaffected), and an
        # 'ambiguous' chunk is exactly the kind of conflict that needs to
        # reach the user/Claude rather than be silently dropped.
        # Exhaustive mode always keeps every match regardless of status,
        # since its purpose is a complete list/audit trail, not "the
        # current rule".
        current_only = [r for r in ordered if r.status != "superseded"]
        ordered = (current_only if current_only else ordered)[:top_n]

    mode = "exhaustive" if exhaustive else "specific"
    return ordered, mode
