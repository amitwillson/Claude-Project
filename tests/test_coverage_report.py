import json
import sqlite3
from pathlib import Path

from storage import db as storedb
from storage.coverage_report import build_report


def _write_index(docs_dir: Path, records: list[dict]) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "_index.json").write_text(json.dumps(records), encoding="utf-8")


def _base_record(doc_id: str, section_path: str, download_ok: bool = True) -> dict:
    return {
        "doc_id": doc_id,
        "section_path": section_path,
        "title": "Some Circular",
        "date_raw": None,
        "date_parsed": None,
        "source_url": "https://example.com/x.pdf",
        "local_path": f"documents/{doc_id}.pdf",
        "sha1": "abc",
        "doc_type": "pdf",
        "menu_id": "1",
        "discovered_at": "2024-01-01T00:00:00+00:00",
        "download_ok": download_ok,
        "http_status": 200,
        "error": None,
    }


def test_build_report_with_no_database_yet(tmp_path):
    docs_dir = tmp_path / "documents"
    _write_index(docs_dir, [_base_record("d1", "A > B")])

    report = build_report(docs_dir, tmp_path / "nonexistent.sqlite3", tmp_path / "coverage_report.md")

    assert "Documents discovered: **1**" in report
    assert "Documents extracted/indexed: **0**" in report
    assert (tmp_path / "coverage_report.md").exists()


def test_build_report_counts_document_number_detection_rate(tmp_path):
    docs_dir = tmp_path / "documents"
    _write_index(docs_dir, [_base_record("d1", "A > B"), _base_record("d2", "A > B")])

    db_path = tmp_path / "test.sqlite3"
    conn = storedb.connect(db_path)
    storedb.upsert_document(
        conn,
        {
            "doc_id": "d1", "title": "T1", "section_path": "A > B", "date_raw": None,
            "date_parsed": None, "source_url": "https://x", "local_path": "documents/d1.pdf",
            "sha1": "a", "doc_type": "pdf", "has_text_layer": 1, "ocr_used": 0,
            "ocr_confidence": None, "word_count": 100, "page_count": 1, "extraction_error": None,
            "needs_review": 0, "indexed_at": "2024-01-01T00:00:00", "document_number": "TC-I/2020/1/1",
        },
    )
    storedb.upsert_document(
        conn,
        {
            "doc_id": "d2", "title": "T2", "section_path": "A > B", "date_raw": None,
            "date_parsed": None, "source_url": "https://x", "local_path": "documents/d2.pdf",
            "sha1": "b", "doc_type": "pdf", "has_text_layer": 1, "ocr_used": 0,
            "ocr_confidence": None, "word_count": 100, "page_count": 1, "extraction_error": None,
            "needs_review": 0, "indexed_at": "2024-01-01T00:00:00", "document_number": None,
        },
    )
    conn.close()

    report = build_report(docs_dir, db_path, tmp_path / "coverage_report.md")

    assert "Documents extracted/indexed: **2**" in report
    assert "detected notification/letter number: **1** (50.0%)" in report


def test_build_report_migrates_a_pre_existing_old_schema_database(tmp_path):
    """Regression test: build_report() must use storage.db.connect() (which
    runs schema migrations), not a raw sqlite3.connect(), or a database
    created before document_number existed would crash this report with
    'no such column: document_number'."""
    docs_dir = tmp_path / "documents"
    _write_index(docs_dir, [_base_record("d1", "A > B")])

    db_path = tmp_path / "old.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.execute(
        """CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY, title TEXT, section_path TEXT, date_raw TEXT,
            date_parsed TEXT, source_url TEXT, local_path TEXT, sha1 TEXT, doc_type TEXT,
            has_text_layer INTEGER, ocr_used INTEGER, ocr_confidence REAL, word_count INTEGER,
            page_count INTEGER, extraction_error TEXT, needs_review INTEGER DEFAULT 0, indexed_at TEXT
        )"""
    )
    old_conn.execute(
        "INSERT INTO documents (doc_id, title, section_path, needs_review, indexed_at) "
        "VALUES ('d1', 'T1', 'A > B', 0, '2024-01-01T00:00:00')"
    )
    old_conn.commit()
    old_conn.close()

    report = build_report(docs_dir, db_path, tmp_path / "coverage_report.md")

    assert "Documents extracted/indexed: **1**" in report
    assert "detected notification/letter number: **0** (0.0%)" in report
