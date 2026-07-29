# Retrieval: specific mode (vector top-N) + exhaustive mode (large top-N +
# FTS5 keyword union). Exhaustive mode triggers on phrasing like "list all",
# "every circular", "summarize all", "all rules on", or an explicit flag.
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

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
        )

    ordered = sorted(results.values(), key=lambda r: r.score, reverse=True)
    if not exhaustive:
        ordered = ordered[:top_n]

    mode = "exhaustive" if exhaustive else "specific"
    return ordered, mode
