# SQLite schema + incremental upsert helpers.
#
# Tables:
#   documents          - one row per source document (from the scraper index
#                         + extraction-quality signals)
#   chunks             - one row per chunk, FTS5-mirrored for keyword search
#   chunks_fts         - FTS5 virtual table mirroring chunks.text
#   supersession_refs  - best-effort supersession/amendment relationships
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    section_path TEXT,
    date_raw TEXT,
    date_parsed TEXT,
    source_url TEXT,
    local_path TEXT,
    sha1 TEXT,
    doc_type TEXT,
    has_text_layer INTEGER,
    ocr_used INTEGER,
    ocr_confidence REAL,
    word_count INTEGER,
    page_count INTEGER,
    extraction_error TEXT,
    needs_review INTEGER DEFAULT 0,
    indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER,
    text TEXT,
    title TEXT,
    section_path TEXT,
    date TEXT,
    source_url TEXT,
    local_path TEXT,
    doc_type TEXT,
    clause_ref TEXT,
    page_start INTEGER,
    page_end INTEGER,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    title,
    content=''
);

CREATE TABLE IF NOT EXISTS supersession_refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    relation TEXT,
    referenced_label TEXT,
    referenced_number TEXT,
    referenced_year TEXT,
    referenced_date_raw TEXT,
    matched_sentence TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Streamlit's UI (qa/webapp.py, qa/dashboard.py)
    # caches this connection with @st.cache_resource across reruns, but
    # Streamlit can execute a rerun on a different internal thread than the
    # one that created the connection -- sqlite3 refuses cross-thread use by
    # default. Safe here since this app never writes concurrently from
    # multiple threads; SQLite itself serializes access within a connection.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS only helps brand-new databases -- an
    existing chunks table (from before clause_ref/page_start/page_end were
    added) needs these columns added in place so already-indexed data isn't
    lost or requires a full re-extraction."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
    for column, coltype in (("clause_ref", "TEXT"), ("page_start", "INTEGER"), ("page_end", "INTEGER")):
        if column not in existing:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {column} {coltype}")
    conn.commit()


def upsert_document(conn: sqlite3.Connection, doc: dict) -> None:
    fields = [
        "doc_id", "title", "section_path", "date_raw", "date_parsed",
        "source_url", "local_path", "sha1", "doc_type", "has_text_layer",
        "ocr_used", "ocr_confidence", "word_count", "page_count",
        "extraction_error", "needs_review", "indexed_at",
    ]
    values = [doc.get(f) for f in fields]
    placeholders = ", ".join("?" for _ in fields)
    updates = ", ".join(f"{f}=excluded.{f}" for f in fields if f != "doc_id")
    conn.execute(
        f"""INSERT INTO documents ({', '.join(fields)}) VALUES ({placeholders})
            ON CONFLICT(doc_id) DO UPDATE SET {updates}""",
        values,
    )
    conn.commit()


def document_exists_with_sha1(conn: sqlite3.Connection, doc_id: str, sha1: Optional[str]) -> bool:
    """Used for incremental extraction: skip re-processing a document whose
    sha1 hasn't changed since it was last indexed AND whose extraction
    previously succeeded cleanly (needs_review=0). A flagged document's file
    hash doesn't change on retry, but the *environment* might (e.g. Poppler
    getting installed) — so flagged documents are always retried rather than
    skipped forever."""
    row = conn.execute(
        "SELECT sha1, needs_review FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return False
    if row["needs_review"]:
        return False
    return sha1 is not None and row["sha1"] == sha1


def delete_chunks_for_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    ids = [r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))]
    for chunk_id in ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, chunks: Iterable) -> None:
    for c in chunks:
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, doc_id, chunk_index, text, title, section_path, date, source_url, local_path, doc_type,
                clause_ref, page_start, page_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                c.chunk_id, c.doc_id, c.chunk_index, c.text, c.title,
                c.section_path, c.date, c.source_url, c.local_path, c.doc_type,
                c.clause_ref, c.page_start, c.page_end,
            ),
        )
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, text, title) VALUES (?, ?, ?)",
            (c.chunk_id, c.text, c.title),
        )
    conn.commit()


def insert_supersession_refs(conn: sqlite3.Connection, doc_id: str, refs: Iterable) -> None:
    conn.execute("DELETE FROM supersession_refs WHERE doc_id = ?", (doc_id,))
    for ref in refs:
        conn.execute(
            """INSERT INTO supersession_refs
               (doc_id, relation, referenced_label, referenced_number, referenced_year, referenced_date_raw, matched_sentence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id, ref.relation, ref.referenced_label, ref.referenced_number,
                ref.referenced_year, ref.referenced_date_raw, ref.matched_sentence,
            ),
        )
    conn.commit()


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[sqlite3.Row]:
    """Full-text keyword search over chunk text via FTS5. Falls back to a
    LIKE search if the query can't be parsed as FTS5 syntax (e.g. contains
    bare punctuation that trips the query parser)."""
    safe_query = _fts_escape(query)
    try:
        return conn.execute(
            """SELECT c.* FROM chunks_fts f
               JOIN chunks c ON c.chunk_id = f.chunk_id
               WHERE chunks_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (safe_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        like_query = f"%{query}%"
        return conn.execute(
            "SELECT * FROM chunks WHERE text LIKE ? OR title LIKE ? LIMIT ?",
            (like_query, like_query, limit),
        ).fetchall()


def _fts_escape(query: str) -> str:
    # Wrap each token in double quotes so FTS5 treats them as literal terms,
    # avoiding syntax errors on punctuation-heavy queries (dates, "No.45", etc).
    tokens = [t for t in query.replace('"', " ").split() if t]
    return " ".join(f'"{t}"' for t in tokens) if tokens else '""'
