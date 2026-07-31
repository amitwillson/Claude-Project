from extraction.backfill_document_ref import run
from storage import db as storedb


def _seed(conn, doc_id: str, chunk_text: str, date_raw="15-08-2019", date_parsed="2019-08-15"):
    storedb.upsert_document(
        conn,
        {
            "doc_id": doc_id,
            "title": "Some Circular",
            "section_path": "A > B",
            "date_raw": date_raw,
            "date_parsed": date_parsed,
            "source_url": "https://example.com/x.pdf",
            "local_path": "documents/x.pdf",
            "sha1": "abc123",
            "doc_type": "pdf",
            "has_text_layer": 1,
            "ocr_used": 0,
            "ocr_confidence": None,
            "word_count": 200,
            "page_count": 2,
            "extraction_error": None,
            "needs_review": 0,
            "indexed_at": "2024-01-01T00:00:00",
            "document_number": None,
        },
    )

    class _C:
        pass

    c = _C()
    c.chunk_id = f"{doc_id}::chunk0"
    c.doc_id = doc_id
    c.chunk_index = 0
    c.text = chunk_text
    c.title = "Some Circular"
    c.section_path = "A > B"
    c.date = date_parsed
    c.source_url = "https://example.com/x.pdf"
    c.local_path = "documents/x.pdf"
    c.doc_type = "pdf"
    c.clause_ref = None
    c.page_start = 1
    c.page_end = 1
    c.circular_number = None
    storedb.insert_chunks(conn, [c])


def test_backfill_updates_document_and_chunks(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed(
        conn,
        "doc1",
        "No. TC-I/2020/109/1\n\nNew Delhi, dated 23.03.2020\n\nSubject: Refund policy.",
        date_raw=None,
        date_parsed=None,
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["documents_checked"] == 1
    assert stats["documents_updated"] == 1

    conn = storedb.connect(db_path)
    doc = conn.execute("SELECT document_number, date_raw, date_parsed FROM documents WHERE doc_id = 'doc1'").fetchone()
    assert doc["document_number"] == "TC-I/2020/109/1"
    assert doc["date_parsed"] == "2020-03-23"

    chunk = conn.execute("SELECT circular_number, date FROM chunks WHERE doc_id = 'doc1'").fetchone()
    assert chunk["circular_number"] == "TC-I/2020/109/1"
    assert chunk["date"] == "2020-03-23"
    conn.close()


def test_backfill_leaves_existing_scraped_date_when_nothing_detected(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed(conn, "doc2", "This document has no recognizable header at all.")
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["documents_checked"] == 1
    assert stats["documents_updated"] == 0

    conn = storedb.connect(db_path)
    doc = conn.execute("SELECT document_number, date_raw, date_parsed FROM documents WHERE doc_id = 'doc2'").fetchone()
    assert doc["document_number"] is None
    assert doc["date_raw"] == "15-08-2019"  # untouched
    assert doc["date_parsed"] == "2019-08-15"  # untouched
    conn.close()


def test_backfill_skips_documents_with_no_chunks(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    storedb.upsert_document(
        conn,
        {
            "doc_id": "doc3", "title": "No chunks yet", "section_path": "A",
            "date_raw": None, "date_parsed": None, "source_url": "https://x",
            "local_path": "documents/x.pdf", "sha1": "x", "doc_type": "pdf",
            "has_text_layer": 0, "ocr_used": 0, "ocr_confidence": None,
            "word_count": 0, "page_count": 0, "extraction_error": "boom",
            "needs_review": 1, "indexed_at": "2024-01-01T00:00:00", "document_number": None,
        },
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["documents_checked"] == 1
    assert stats["documents_updated"] == 0
