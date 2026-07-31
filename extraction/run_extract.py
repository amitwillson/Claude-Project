# CLI: python -m extraction.run_extract
#
# Iterates documents/_index.json, extracts text (PDF w/ OCR fallback, or
# Office formats), chunks, detects supersession language, and writes
# everything into the SQLite store + vector store. Incremental: skips
# documents whose sha1 hasn't changed since the last run. Emits
# needs_manual_review.md per addendum A.
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from extraction.chunker import chunk_text
from extraction.document_ref import detect_document_ref
from extraction.office_extract import extract_office_text
from extraction.pdf_extract import extract_pdf_text, flag_extraction_outliers
from extraction.supersession import find_supersession_refs
from scraper.models import load_index
from storage import db as storedb

logger = logging.getLogger("extraction.run_extract")


def extract_document(local_path: Path, doc_type: str):
    """Returns (text, quality_dict) for a single document. quality_dict["pages"]
    is a list of per-page text when available (PDFs), enabling clause/page
    citation in chunking; falls back to None for formats without page
    boundaries (chunk_text treats a plain string as a single page)."""
    if doc_type == "pdf":
        result = extract_pdf_text(local_path)
        needs_review = flag_extraction_outliers(result)
        return result.text, {
            "has_text_layer": result.has_text_layer,
            "ocr_used": result.ocr_used,
            "ocr_confidence": result.ocr_confidence,
            "word_count": result.word_count,
            "page_count": result.page_count,
            "extraction_error": result.error,
            "needs_review": needs_review,
            "pages": result.pages,
        }

    if doc_type in ("docx", "xlsx", "xls", "doc"):
        result = extract_office_text(local_path)
        needs_review = bool(result.error) or result.best_effort or result.word_count == 0
        return result.text, {
            "has_text_layer": not result.best_effort,
            "ocr_used": False,
            "ocr_confidence": None,
            "word_count": result.word_count,
            "page_count": None,
            "extraction_error": result.error,
            "needs_review": needs_review,
            "pages": None,
        }

    return "", {
        "has_text_layer": False,
        "ocr_used": False,
        "ocr_confidence": None,
        "word_count": 0,
        "page_count": None,
        "extraction_error": f"unsupported doc_type: {doc_type}",
        "needs_review": True,
        "pages": None,
    }


def run(
    docs_dir: str, db_path: str, vector_dir: str, skip_embeddings: bool = False, force: bool = False
) -> dict:
    records = load_index(Path(docs_dir))
    conn = storedb.connect(db_path)

    vector_store = None
    embedder = None
    if not skip_embeddings:
        try:
            from storage.embeddings import get_default_provider
            from storage.vectorstore import VectorStore

            embedder = get_default_provider()
            vector_store = VectorStore(vector_dir)
        except Exception as exc:  # sentence-transformers/chromadb not installed
            logger.warning("Embeddings/vector store unavailable (%s) — skipping semantic indexing.", exc)

    review_entries: list[str] = []
    processed = 0
    skipped = 0

    for rec in records:
        if not rec.download_ok or not rec.local_path:
            continue
        local_path = Path(rec.local_path)
        if not local_path.exists():
            continue

        if not force and storedb.document_exists_with_sha1(conn, rec.doc_id, rec.sha1):
            skipped += 1
            continue

        text, quality = extract_document(local_path, rec.doc_type)
        processed += 1

        # The document's own letterhead/reference line (read directly from
        # its extracted text) is a more authoritative source for the
        # notification/letter number and issue date than the scraped
        # listing-page metadata -- prefer it when detected, fall back to the
        # scraped values otherwise. See extraction/document_ref.py.
        ref = detect_document_ref(text)
        date_raw = ref.date_raw or rec.date_raw
        date_parsed = ref.date_parsed or rec.date_parsed

        doc_row = {
            "doc_id": rec.doc_id,
            "title": rec.title,
            "section_path": rec.section_path,
            "date_raw": date_raw,
            "date_parsed": date_parsed,
            "source_url": rec.source_url,
            "local_path": rec.local_path,
            "sha1": rec.sha1,
            "doc_type": rec.doc_type,
            "has_text_layer": int(quality["has_text_layer"]),
            "ocr_used": int(quality["ocr_used"]),
            "ocr_confidence": quality["ocr_confidence"],
            "word_count": quality["word_count"],
            "page_count": quality["page_count"],
            "extraction_error": quality["extraction_error"],
            "needs_review": int(quality["needs_review"]),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "document_number": ref.number,
        }
        storedb.upsert_document(conn, doc_row)

        if quality["needs_review"]:
            reason = quality["extraction_error"] or "low word count / possible extraction loss"
            review_entries.append(
                f"- **{rec.title}** ({rec.section_path}) — `{rec.local_path}` — {reason}"
            )

        storedb.delete_chunks_for_doc(conn, rec.doc_id)

        # A document whose text couldn't be extracted at all (OCR genuinely
        # failed, corrupt file, etc.) must still be indexable by its
        # metadata -- otherwise it's invisible to Q&A, and a user asking
        # about that exact circular gets told "no matching circular found"
        # even though the PDF genuinely exists. Index a placeholder chunk
        # that says so explicitly, so retrieval can still surface it and
        # point back to the original PDF rather than silently omitting it.
        is_placeholder = not text.strip()
        indexable_text = quality.get("pages") or text
        if is_placeholder:
            indexable_text = (
                f"[No machine-readable text could be extracted from this document by OCR. "
                f"Title: {rec.title}. Section: {rec.section_path}. "
                f"Date: {rec.date_parsed or rec.date_raw or 'unknown'}. "
                f"Refer to the original PDF ({rec.local_path}) for its content.]"
            )

        chunks = chunk_text(
            text=indexable_text,
            doc_id=rec.doc_id,
            section_path=rec.section_path,
            title=rec.title,
            date=date_parsed or date_raw,
            source_url=rec.source_url,
            local_path=rec.local_path,
            doc_type=rec.doc_type,
            circular_number=ref.number,
        )
        storedb.insert_chunks(conn, chunks)

        if not is_placeholder:
            refs = find_supersession_refs(text)
            storedb.insert_supersession_refs(conn, rec.doc_id, refs)

        if vector_store is not None and embedder is not None and chunks:
            embeddings = embedder.embed([c.text for c in chunks])
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "section_path": c.section_path,
                    "date": c.date or "",
                    "source_url": c.source_url,
                    "local_path": c.local_path or "",
                    "chunk_index": c.chunk_index,
                    "clause_ref": c.clause_ref or "",
                    "page_start": c.page_start or 0,
                    "page_end": c.page_end or 0,
                    "circular_number": c.circular_number or "",
                }
                for c in chunks
            ]
            vector_store.add_chunks(
                chunk_ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=metadatas,
            )

        logger.info(
            "Extracted %s (%d words, needs_review=%s)",
            rec.local_path, quality["word_count"], quality["needs_review"],
        )

    _write_manual_review_report(review_entries)

    conn.close()
    return {"processed": processed, "skipped": skipped, "flagged": len(review_entries)}


def _write_manual_review_report(entries: list[str], out_path: str = "needs_manual_review.md") -> None:
    lines = ["# Documents Needing Manual Review", ""]
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    if not entries:
        lines.append("No documents currently flagged for manual review.")
    else:
        lines.append(
            f"{len(entries)} document(s) flagged (extraction failure, empty text, "
            "or suspiciously low word count relative to page count):"
        )
        lines.append("")
        lines.extend(entries)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract, chunk, and index downloaded circulars.")
    parser.add_argument("--docs-dir", default="documents")
    parser.add_argument("--db-path", default="data/ir_kb.sqlite3")
    parser.add_argument("--vector-dir", default="data/chroma_db")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding/vector indexing (metadata+FTS only).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract and re-chunk every document, ignoring the incremental sha1/needs_review skip. "
        "Needed after a chunking-logic change (e.g. clause/page tracking) so already-successful "
        "documents get re-chunked with the new logic, not just newly-added/previously-flagged ones. "
        "Slow: re-runs OCR for every document that originally needed it.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stats = run(
        args.docs_dir, args.db_path, args.vector_dir, skip_embeddings=args.skip_embeddings, force=args.force
    )
    logger.info("Extraction complete: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
