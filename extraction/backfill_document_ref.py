# Backfill the notification/letter number + in-document date for documents
# that were already successfully extracted BEFORE extraction/document_ref.py
# existed -- without re-running extraction or OCR.
#
# Why this is safe to skip re-extraction: a document's letterhead/reference
# line is always within the first few hundred words, well inside its first
# chunk (chunk_index=0), which is already stored from the prior extraction
# run. Re-detecting from that existing text is enough; there is no need to
# touch the source PDF, let alone redo OCR, for documents that already
# extracted cleanly.
#
# Usage: python -m extraction.backfill_document_ref
from __future__ import annotations

import argparse
import logging

from extraction.document_ref import detect_document_ref
from storage import db as storedb

logger = logging.getLogger("extraction.backfill_document_ref")


def run(db_path: str, vector_dir: str, skip_vector: bool = False) -> dict:
    conn = storedb.connect(db_path)

    vector_store = None
    if not skip_vector:
        try:
            from storage.vectorstore import VectorStore

            vector_store = VectorStore(vector_dir)
        except Exception as exc:
            logger.warning(
                "Vector store unavailable (%s) -- updating SQLite only; "
                "Chroma metadata will lag until the next full extraction run.",
                exc,
            )

    doc_ids = [r["doc_id"] for r in conn.execute("SELECT doc_id FROM documents")]
    checked = 0
    updated = 0

    for doc_id in doc_ids:
        checked += 1
        first_chunk = conn.execute(
            "SELECT text FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC LIMIT 1",
            (doc_id,),
        ).fetchone()
        if first_chunk is None:
            continue  # document has no chunks yet (never extracted, or placeholder-only)

        ref = detect_document_ref(first_chunk["text"])
        if ref.number is None and ref.date_parsed is None:
            continue  # nothing new detected -- leave existing data untouched

        row = conn.execute(
            "SELECT date_raw, date_parsed FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        new_date_raw = ref.date_raw or row["date_raw"]
        new_date_parsed = ref.date_parsed or row["date_parsed"]

        conn.execute(
            "UPDATE documents SET document_number = ?, date_raw = ?, date_parsed = ? WHERE doc_id = ?",
            (ref.number, new_date_raw, new_date_parsed, doc_id),
        )

        chunk_ids = [
            r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
        ]
        conn.executemany(
            "UPDATE chunks SET circular_number = ?, date = ? WHERE chunk_id = ?",
            [(ref.number, new_date_parsed or new_date_raw, cid) for cid in chunk_ids],
        )
        conn.commit()

        if vector_store is not None and chunk_ids:
            try:
                vector_store.update_metadata(
                    chunk_ids,
                    {"circular_number": ref.number or "", "date": new_date_parsed or new_date_raw or ""},
                )
            except Exception as exc:
                logger.warning("Vector metadata update failed for %s: %s", doc_id, exc)

        updated += 1
        logger.info("Backfilled %s: number=%s date=%s", doc_id, ref.number, new_date_parsed or new_date_raw)

    conn.close()
    return {"documents_checked": checked, "documents_updated": updated}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill notification/letter number + in-document date from already-extracted "
        "chunk text, without re-running extraction or OCR."
    )
    parser.add_argument("--db-path", default="data/ir_kb.sqlite3")
    parser.add_argument("--vector-dir", default="data/chroma_db")
    parser.add_argument("--skip-vector", action="store_true", help="Update SQLite only, skip Chroma metadata.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stats = run(args.db_path, args.vector_dir, skip_vector=args.skip_vector)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
