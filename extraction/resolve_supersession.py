# Resolves the best-effort supersession references detected by
# extraction/supersession.py (which only captures the *text* of a reference
# -- "in supersession of Circular No. X dated Y" -- found inside a document)
# into an actual link between two INDEXED documents, so retrieval/answering
# can tell the old, no-longer-current circular apart from its replacement.
#
# Why this is a separate step from extraction/supersession.py: a document
# can only be marked "superseded by Z" once Z itself has been indexed, which
# may happen in a later scraper/extraction run than the document it
# supersedes. Re-running this script (safe/idempotent, no PDF/API access
# needed -- same design as extraction/backfill_document_ref.py) re-resolves
# against whatever is currently indexed.
#
# Matching is intentionally conservative: it only resolves a reference when
# the referenced number matches an indexed document's own detected
# document_number (extraction/document_ref.py) after loose normalization
# (case/punctuation-insensitive). No fuzzy title matching -- reference-number
# formats are inconsistent enough (see extraction/supersession.py) without
# adding a second, noisier layer of guessing.
#
# Usage: python -m extraction.resolve_supersession
from __future__ import annotations

import argparse
import logging
import re

from storage import db as storedb

logger = logging.getLogger("extraction.resolve_supersession")


def _normalize_number(number: str | None) -> str | None:
    if not number:
        return None
    normalized = re.sub(r"[^A-Za-z0-9]", "", number).lower()
    return normalized or None


def _format_summary(title: str, document_number: str | None, date_parsed: str | None) -> str:
    parts = [title or "an indexed circular"]
    if document_number:
        parts.append(f"(No. {document_number})")
    if date_parsed:
        parts.append(f"dated {date_parsed}")
    return " ".join(parts)


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

    documents = {
        row["doc_id"]: row
        for row in conn.execute(
            "SELECT doc_id, title, document_number, date_parsed, superseded_by_doc_id FROM documents"
        )
    }
    by_normalized_number: dict[str, str] = {}
    for doc_id, row in documents.items():
        norm = _normalize_number(row["document_number"])
        if norm:
            by_normalized_number[norm] = doc_id

    refs = conn.execute(
        "SELECT doc_id, referenced_number, referenced_date_raw FROM supersession_refs"
    ).fetchall()

    stats = {"refs_checked": 0, "resolved": 0, "unmatched": 0, "skipped_date_order": 0}
    # Track the winning (most-recent) superseder chosen so far per target doc
    # in this run, so multiple refs pointing at the same old document don't
    # flip-flop -- the target ends up pointing at whichever superseding
    # document has the latest date_parsed.
    winners: dict[str, tuple[str, str]] = {}  # target_doc_id -> (superseder_doc_id, superseder_date_parsed)

    for row in refs:
        stats["refs_checked"] += 1
        superseder_id = row["doc_id"]
        target_id = by_normalized_number.get(_normalize_number(row["referenced_number"]))

        if target_id is None or target_id == superseder_id:
            stats["unmatched"] += 1
            continue

        superseder = documents.get(superseder_id)
        target = documents.get(target_id)
        if superseder is None or target is None:
            stats["unmatched"] += 1
            continue

        superseder_date = superseder["date_parsed"]
        target_date = target["date_parsed"]
        if superseder_date and target_date and superseder_date < target_date:
            # The doc claiming to supersede something is dated *before* the
            # target -- almost certainly a mismatched number, not a real
            # supersession. Skip rather than record a nonsensical link.
            stats["skipped_date_order"] += 1
            continue

        current_winner = winners.get(target_id)
        if current_winner is not None:
            _, current_date = current_winner
            if current_date and superseder_date and superseder_date <= current_date:
                continue  # an already-chosen, more-recent superseder wins
        winners[target_id] = (superseder_id, superseder_date or "")

    for target_id, (superseder_id, _) in winners.items():
        superseder = documents[superseder_id]
        summary = _format_summary(superseder["title"], superseder["document_number"], superseder["date_parsed"])
        storedb.set_superseded(conn, target_id, superseder_id, summary)
        stats["resolved"] += 1

        if vector_store is not None:
            chunk_ids = [r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (target_id,))]
            if chunk_ids:
                try:
                    vector_store.update_metadata(
                        chunk_ids,
                        {"superseded_by_doc_id": superseder_id, "superseded_by_summary": summary},
                    )
                except Exception as exc:
                    logger.warning("Vector metadata update failed for %s: %s", target_id, exc)

        logger.info("Resolved: %s is superseded by %s (%s)", target_id, superseder_id, summary)

    conn.close()
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve detected supersession references into links between indexed documents."
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
