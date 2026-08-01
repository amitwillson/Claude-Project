from qa.retrieval import retrieve
from storage import db as storedb


def _chunk(chunk_id, doc_id, text, title="Refund policy", superseded_by_doc_id=""):
    class _C:
        pass

    c = _C()
    c.chunk_id = chunk_id
    c.doc_id = doc_id
    c.chunk_index = 0
    c.text = text
    c.title = title
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


def _seed_doc(conn, doc_id):
    storedb.upsert_document(
        conn,
        {
            "doc_id": doc_id, "title": "Refund policy", "section_path": "A > B",
            "date_raw": "2020-01-01", "date_parsed": "2020-01-01", "source_url": "https://x",
            "local_path": "documents/x.pdf", "sha1": doc_id, "doc_type": "pdf", "has_text_layer": 1,
            "ocr_used": 0, "ocr_confidence": None, "word_count": 10, "page_count": 1,
            "extraction_error": None, "needs_review": 0, "indexed_at": "2024-01-01T00:00:00",
            "document_number": None,
        },
    )


def test_specific_mode_filters_out_superseded_chunk_when_a_current_one_exists(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old")
    _seed_doc(conn, "new")
    storedb.insert_chunks(conn, [_chunk("old::0", "old", "refund of unused tickets old rule")])
    storedb.insert_chunks(conn, [_chunk("new::0", "new", "refund of unused tickets new rule")])
    storedb.set_superseded(conn, "old", "new", "Refund policy (No. TC-I/2021/1) dated 2021-01-01")

    chunks, mode = retrieve("refund of unused tickets", conn, exhaustive=False)
    assert mode == "specific"
    assert "old::0" not in [c.chunk_id for c in chunks]
    assert "new::0" in [c.chunk_id for c in chunks]
    conn.close()


def test_specific_mode_falls_back_to_superseded_chunk_when_nothing_else_matches(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old")
    storedb.insert_chunks(conn, [_chunk("old::0", "old", "refund of unused tickets old rule")])
    storedb.set_superseded(conn, "old", "not_indexed_successor", "some circular")

    chunks, mode = retrieve("refund of unused tickets", conn, exhaustive=False)
    assert "old::0" in [c.chunk_id for c in chunks]
    conn.close()


def test_exhaustive_mode_keeps_superseded_chunks(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old")
    _seed_doc(conn, "new")
    storedb.insert_chunks(conn, [_chunk("old::0", "old", "refund of unused tickets old rule")])
    storedb.insert_chunks(conn, [_chunk("new::0", "new", "refund of unused tickets new rule")])
    storedb.set_superseded(conn, "old", "new", "Refund policy (No. TC-I/2021/1) dated 2021-01-01")

    chunks, mode = retrieve("refund of unused tickets", conn, exhaustive=True)
    assert mode == "exhaustive"
    ids = [c.chunk_id for c in chunks]
    assert "old::0" in ids and "new::0" in ids
    conn.close()
