from extraction.resolve_supersession import run
from extraction.supersession import SupersessionRef
from storage import db as storedb


def _seed_doc(conn, doc_id, title, document_number, date_parsed, chunk_text="text"):
    storedb.upsert_document(
        conn,
        {
            "doc_id": doc_id, "title": title, "section_path": "A > B",
            "date_raw": date_parsed, "date_parsed": date_parsed,
            "source_url": f"https://example.com/{doc_id}.pdf", "local_path": f"documents/{doc_id}.pdf",
            "sha1": doc_id, "doc_type": "pdf", "has_text_layer": 1, "ocr_used": 0,
            "ocr_confidence": None, "word_count": 100, "page_count": 1,
            "extraction_error": None, "needs_review": 0, "indexed_at": "2024-01-01T00:00:00",
            "document_number": document_number,
        },
    )

    class _C:
        pass

    c = _C()
    c.chunk_id = f"{doc_id}::chunk0"
    c.doc_id = doc_id
    c.chunk_index = 0
    c.text = chunk_text
    c.title = title
    c.section_path = "A > B"
    c.date = date_parsed
    c.source_url = f"https://example.com/{doc_id}.pdf"
    c.local_path = f"documents/{doc_id}.pdf"
    c.doc_type = "pdf"
    c.clause_ref = None
    c.page_start = 1
    c.page_end = 1
    c.circular_number = document_number
    storedb.insert_chunks(conn, [c])


def test_resolves_link_when_referenced_number_matches_indexed_document(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Refund of unused tickets (old)", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "new", "Refund of unused tickets (revised)", "TC-I/2021/9", "2021-06-01")
    storedb.insert_supersession_refs(
        conn, "new",
        [SupersessionRef("supersession", "Circular", "TC-I/2019/1", "2019", "01.01.2019", "in supersession of TC-I/2019/1")],
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["resolved"] == 1

    conn = storedb.connect(db_path)
    doc = conn.execute("SELECT superseded_by_doc_id, superseded_by_summary FROM documents WHERE doc_id = 'old'").fetchone()
    assert doc["superseded_by_doc_id"] == "new"
    assert "TC-I/2021/9" in doc["superseded_by_summary"]

    chunk = conn.execute("SELECT superseded_by_doc_id FROM chunks WHERE doc_id = 'old'").fetchone()
    assert chunk["superseded_by_doc_id"] == "new"

    # the successor itself must not be marked as superseded
    new_doc = conn.execute("SELECT superseded_by_doc_id FROM documents WHERE doc_id = 'new'").fetchone()
    assert new_doc["superseded_by_doc_id"] is None
    conn.close()


def test_number_match_tolerates_punctuation_differences(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Old rule", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "new", "New rule", "TC-I/2021/9", "2021-06-01")
    storedb.insert_supersession_refs(
        conn, "new",
        [SupersessionRef("supersession", "Circular", "TC I 2019 1", "2019", None, "in supersession of TC I 2019 1")],
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["resolved"] == 1


def test_skips_when_no_indexed_document_matches_the_reference(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "new", "New rule", "TC-I/2021/9", "2021-06-01")
    storedb.insert_supersession_refs(
        conn, "new",
        [SupersessionRef("supersession", "Circular", "Unindexed/1999/1", "1999", None, "in supersession of Unindexed/1999/1")],
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["resolved"] == 0
    assert stats["unmatched"] == 1


def test_rejects_link_when_superseder_is_dated_before_target(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Old rule", "TC-I/2021/1", "2021-01-01")
    _seed_doc(conn, "new", "Mismatched number, earlier date", "TC-I/2019/9", "2019-06-01")
    storedb.insert_supersession_refs(
        conn, "new",
        [SupersessionRef("supersession", "Circular", "TC-I/2021/1", "2021", None, "in supersession of TC-I/2021/1")],
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["resolved"] == 0
    assert stats["skipped_date_order"] == 1


def test_most_recent_superseder_wins_when_multiple_reference_the_same_document(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Old rule", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "mid", "Mid revision", "TC-I/2020/1", "2020-01-01")
    _seed_doc(conn, "latest", "Latest revision", "TC-I/2022/1", "2022-01-01")
    storedb.insert_supersession_refs(
        conn, "mid",
        [SupersessionRef("supersession", "Circular", "TC-I/2019/1", "2019", None, "s1")],
    )
    storedb.insert_supersession_refs(
        conn, "latest",
        [SupersessionRef("supersession", "Circular", "TC-I/2019/1", "2019", None, "s2")],
    )
    conn.close()

    run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)

    conn = storedb.connect(db_path)
    doc = conn.execute("SELECT superseded_by_doc_id FROM documents WHERE doc_id = 'old'").fetchone()
    assert doc["superseded_by_doc_id"] == "latest"
    conn.close()
