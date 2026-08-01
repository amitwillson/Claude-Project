from storage import db as storedb


def _chunk(chunk_id="c1", doc_id="d1", text="refund of unused tickets policy"):
    class _C:
        pass

    c = _C()
    c.chunk_id = chunk_id
    c.doc_id = doc_id
    c.chunk_index = 0
    c.text = text
    c.title = "Refund policy"
    c.section_path = "A > B"
    c.date = "2020-01-01"
    c.source_url = "https://x"
    c.local_path = "documents/x.pdf"
    c.doc_type = "pdf"
    c.clause_ref = None
    c.page_start = 1
    c.page_end = 1
    c.circular_number = None
    return c


def test_fts_search_finds_inserted_chunk(tmp_path):
    conn = storedb.connect(tmp_path / "test.sqlite3")
    storedb.insert_chunks(conn, [_chunk()])
    results = storedb.fts_search(conn, "refund of unused tickets")
    assert [r["chunk_id"] for r in results] == ["c1"]
    conn.close()


def test_migrate_rebuilds_a_pre_existing_contentless_fts_table(tmp_path):
    """Regression test for a bug where chunks_fts was created with
    content='' (FTS5 'contentless' mode), which never stores column values --
    chunk_id always came back NULL, silently breaking every keyword search.
    A database created before the fix must be rebuilt in place on connect()."""
    db_path = tmp_path / "test.sqlite3"
    import sqlite3

    raw = sqlite3.connect(str(db_path))
    raw.execute(
        """CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, text TEXT,
            title TEXT, section_path TEXT, date TEXT, source_url TEXT, local_path TEXT,
            doc_type TEXT, clause_ref TEXT, page_start INTEGER, page_end INTEGER, circular_number TEXT
        )"""
    )
    raw.execute(
        "INSERT INTO chunks (chunk_id, doc_id, text, title) VALUES ('c1', 'd1', 'refund of unused tickets', 'Refund policy')"
    )
    raw.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text, title, content='')"
    )
    raw.execute("INSERT INTO chunks_fts (chunk_id, text, title) VALUES ('c1', 'refund of unused tickets', 'Refund policy')")
    raw.commit()
    raw.close()

    conn = storedb.connect(db_path)  # triggers _migrate_fts
    results = storedb.fts_search(conn, "refund of unused tickets")
    assert [r["chunk_id"] for r in results] == ["c1"]
    conn.close()
