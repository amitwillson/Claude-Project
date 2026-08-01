from extraction.resolve_supersession import run
from extraction.supersession import SupersessionRef
from storage import db as storedb


def _seed_doc(conn, doc_id, title, document_number, date_parsed):
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


def _seed_chunk(conn, doc_id, chunk_index, clause_ref, text="text", title="Some Circular"):
    class _C:
        pass

    c = _C()
    c.chunk_id = f"{doc_id}::chunk{chunk_index}"
    c.doc_id = doc_id
    c.chunk_index = chunk_index
    c.text = text
    c.title = title
    c.section_path = "A > B"
    c.date = None
    c.source_url = f"https://example.com/{doc_id}.pdf"
    c.local_path = f"documents/{doc_id}.pdf"
    c.doc_type = "pdf"
    c.clause_ref = clause_ref
    c.page_start = 1
    c.page_end = 1
    c.circular_number = None
    storedb.insert_chunks(conn, [c])
    return c.chunk_id


def _ref(doc_id, relation, referenced_number, referenced_clause=None, referenced_date_raw=None):
    return SupersessionRef(
        relation=relation, referenced_label="Circular", referenced_number=referenced_number,
        referenced_year=None, referenced_date_raw=referenced_date_raw,
        matched_sentence="x", referenced_clause=referenced_clause,
    )


def test_full_supersession_marks_every_chunk_of_target_and_rolls_up_to_document(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Refund policy (old)", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "new", "Refund policy (revised)", "TC-I/2021/9", "2021-06-01")
    c1 = _seed_chunk(conn, "old", 0, "1")
    c2 = _seed_chunk(conn, "old", 1, "2")
    storedb.insert_supersession_refs(conn, "new", [_ref("new", "supersession", "TC-I/2019/1")])
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["chunks_superseded"] == 2
    assert stats["documents_fully_superseded"] == 1

    conn = storedb.connect(db_path)
    for cid in (c1, c2):
        row = conn.execute("SELECT status, status_note FROM chunks WHERE chunk_id = ?", (cid,)).fetchone()
        assert row["status"] == "superseded"
        assert "TC-I/2021/9" in row["status_note"]
    doc = conn.execute("SELECT superseded_by_doc_id FROM documents WHERE doc_id = 'old'").fetchone()
    assert doc["superseded_by_doc_id"] == "new"
    conn.close()


def test_partial_modification_only_marks_the_matching_clause(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Refund policy (old)", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "amend", "Amendment to refund policy", "TC-I/2021/2", "2021-03-01")
    clause1 = _seed_chunk(conn, "old", 0, "1")
    clause4 = _seed_chunk(conn, "old", 1, "4")
    clause4_1 = _seed_chunk(conn, "old", 2, "4.1")  # sub-clause of 4 -- should also match
    storedb.insert_supersession_refs(
        conn, "amend", [_ref("amend", "partial_modification", "TC-I/2019/1", referenced_clause="4")]
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["chunks_amended"] == 2
    assert stats["documents_fully_superseded"] == 0  # doc as a whole is still in force

    conn = storedb.connect(db_path)
    row1 = conn.execute("SELECT status FROM chunks WHERE chunk_id = ?", (clause1,)).fetchone()
    assert row1["status"] == "current"  # untouched clause stays current

    row4 = conn.execute("SELECT status, status_note FROM chunks WHERE chunk_id = ?", (clause4,)).fetchone()
    assert row4["status"] == "amended"
    assert "Clause 4" in row4["status_note"]

    row4_1 = conn.execute("SELECT status FROM chunks WHERE chunk_id = ?", (clause4_1,)).fetchone()
    assert row4_1["status"] == "amended"  # sub-clause of the amended clause

    doc = conn.execute("SELECT superseded_by_doc_id FROM documents WHERE doc_id = 'old'").fetchone()
    assert doc["superseded_by_doc_id"] is None  # never touched -- only chunk-level status changed
    conn.close()


def test_amendment_with_no_detected_clause_is_left_untouched_not_applied_to_whole_document(tmp_path):
    """Regression test: an earlier version of this resolver treated every
    relation as a full supersession, which would have wrongly wiped out an
    entire document just because *some* amendment referencing it existed,
    even with no clause named. It must now be a no-op."""
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Refund policy (old)", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "amend", "Amendment to refund policy", "TC-I/2021/2", "2021-03-01")
    clause1 = _seed_chunk(conn, "old", 0, "1")
    storedb.insert_supersession_refs(conn, "amend", [_ref("amend", "amendment", "TC-I/2019/1")])
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["chunks_amended"] == 0
    assert stats["chunks_superseded"] == 0
    assert stats["skipped_unscoped_partial"] == 1

    conn = storedb.connect(db_path)
    row = conn.execute("SELECT status FROM chunks WHERE chunk_id = ?", (clause1,)).fetchone()
    assert row["status"] == "current"
    conn.close()


def test_conflicting_full_and_partial_signals_on_same_chunk_become_ambiguous(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Refund policy (old)", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "full", "Full reissue", "TC-I/2022/1", "2022-01-01")
    _seed_doc(conn, "amend", "Amendment", "TC-I/2021/2", "2021-01-01")
    clause4 = _seed_chunk(conn, "old", 0, "4")
    storedb.insert_supersession_refs(conn, "full", [_ref("full", "supersession", "TC-I/2019/1")])
    storedb.insert_supersession_refs(
        conn, "amend", [_ref("amend", "partial_modification", "TC-I/2019/1", referenced_clause="4")]
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["chunks_ambiguous"] == 1
    assert stats["documents_fully_superseded"] == 0  # ambiguity blocks the whole-doc rollup too

    conn = storedb.connect(db_path)
    row = conn.execute("SELECT status, status_note FROM chunks WHERE chunk_id = ?", (clause4,)).fetchone()
    assert row["status"] == "ambiguous"
    assert "TC-I/2022/1" in row["status_note"] and "TC-I/2021/2" in row["status_note"]
    conn.close()


def test_full_supersession_incidental_clause_mention_is_ignored_as_scoping(tmp_path):
    """A "Clause" word appearing near a full 'in supersession of' phrase for
    unrelated reasons must not suppress the full-document resolution --
    only PARTIAL_RELATIONS are clause-scoped."""
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Refund policy (old)", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "new", "Refund policy (revised)", "TC-I/2021/9", "2021-06-01")
    clause1 = _seed_chunk(conn, "old", 0, "1")
    storedb.insert_supersession_refs(
        conn, "new",
        [_ref("new", "supersession", "TC-I/2019/1", referenced_clause="5")],  # incidental match, must be ignored
    )
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["chunks_superseded"] == 1

    conn = storedb.connect(db_path)
    row = conn.execute("SELECT status FROM chunks WHERE chunk_id = ?", (clause1,)).fetchone()
    assert row["status"] == "superseded"
    conn.close()


def test_continuation_relation_does_not_change_status(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Some circular", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "cont", "Continuation letter", "TC-I/2019/5", "2019-03-01")
    clause1 = _seed_chunk(conn, "old", 0, "1")
    storedb.insert_supersession_refs(conn, "cont", [_ref("cont", "continuation", "TC-I/2019/1")])
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["informational_skipped"] == 1

    conn = storedb.connect(db_path)
    row = conn.execute("SELECT status FROM chunks WHERE chunk_id = ?", (clause1,)).fetchone()
    assert row["status"] == "current"
    conn.close()


def test_skips_when_no_indexed_document_matches_the_reference(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "new", "New rule", "TC-I/2021/9", "2021-06-01")
    storedb.insert_supersession_refs(conn, "new", [_ref("new", "supersession", "Unindexed/1999/1")])
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["chunks_superseded"] == 0
    assert stats["unmatched"] == 1


def test_rejects_link_when_superseder_is_dated_before_target(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Old rule", "TC-I/2021/1", "2021-01-01")
    _seed_doc(conn, "new", "Mismatched number, earlier date", "TC-I/2019/9", "2019-06-01")
    clause1 = _seed_chunk(conn, "old", 0, "1")
    storedb.insert_supersession_refs(conn, "new", [_ref("new", "supersession", "TC-I/2021/1")])
    conn.close()

    stats = run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)
    assert stats["chunks_superseded"] == 0
    assert stats["skipped_date_order"] == 1

    conn = storedb.connect(db_path)
    row = conn.execute("SELECT status FROM chunks WHERE chunk_id = ?", (clause1,)).fetchone()
    assert row["status"] == "current"
    conn.close()


def test_most_recent_superseder_wins_when_multiple_fully_supersede_the_same_document(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    _seed_doc(conn, "old", "Old rule", "TC-I/2019/1", "2019-01-01")
    _seed_doc(conn, "mid", "Mid revision", "TC-I/2020/1", "2020-01-01")
    _seed_doc(conn, "latest", "Latest revision", "TC-I/2022/1", "2022-01-01")
    clause1 = _seed_chunk(conn, "old", 0, "1")
    storedb.insert_supersession_refs(conn, "mid", [_ref("mid", "supersession", "TC-I/2019/1")])
    storedb.insert_supersession_refs(conn, "latest", [_ref("latest", "supersession", "TC-I/2019/1")])
    conn.close()

    run(str(db_path), vector_dir=str(tmp_path / "chroma"), skip_vector=True)

    conn = storedb.connect(db_path)
    doc = conn.execute("SELECT superseded_by_doc_id FROM documents WHERE doc_id = 'old'").fetchone()
    assert doc["superseded_by_doc_id"] == "latest"
    row = conn.execute("SELECT status_note FROM chunks WHERE chunk_id = ?", (clause1,)).fetchone()
    assert "TC-I/2022/1" in row["status_note"]
    conn.close()
