# Resolves the best-effort supersession/amendment references detected by
# extraction/supersession.py (which only captures the *text* of a reference
# -- "in supersession of Circular No. X dated Y", or "in partial
# modification of Clause 4 of Circular No. X..." -- found inside a
# document) into actual per-CHUNK status on indexed documents, so
# retrieval/answering can tell an old, fully-replaced rule apart from one
# that's only had a single clause amended (whose other clauses are still
# in force) apart from one that's still simply current.
#
# Why per-chunk, not per-document: a document can be superseded in whole
# (extraction.supersession.FULL_RELATIONS -- "in supersession of...") or
# only in part (PARTIAL_RELATIONS -- "in partial modification of Clause 4
# of...", "amends..."). Treating every relation as a full replacement (an
# earlier version of this module did) is a real correctness bug: it would
# hide an entire circular's clauses 1-3 from retrieval just because clause
# 4 was later amended. Each chunk gets its own status instead:
#   'current'    - unaffected, this is still what applies
#   'superseded' - a later document fully replaced the one this chunk is in
#   'amended'    - a later document changed *this specific clause*; other
#                  chunks of the same document may still be 'current'
#   'ambiguous'  - conflicting signals resolved to this chunk (e.g. one
#                  document's text claims to fully supersede the document
#                  this chunk is in, while another, differently-dated
#                  document claims to have only amended this exact clause)
#                  -- deliberately NOT auto-resolved one way or the other;
#                  surfaced to the user instead of silently guessed.
#
# Why this is a separate step from extraction/supersession.py: a document
# can only be resolved against documents that have themselves already been
# indexed, which may happen in a later scraper/extraction run. Re-running
# this script is safe/idempotent (no PDF/API access needed -- same design
# as extraction/backfill_document_ref.py) and re-resolves against whatever
# is currently indexed.
#
# Matching is intentionally conservative: a reference is only resolved
# against an indexed document when the referenced number matches that
# document's own detected document_number (extraction/document_ref.py)
# after loose normalization (case/punctuation-insensitive), and a detected
# clause is only applied to chunks whose own detected clause_ref
# (extraction/chunker.py) matches it (exact, or a sub-clause of it, e.g. a
# reference to "Clause 4" matches a chunk tagged "4.1"). No fuzzy title
# matching -- reference-number and clause-numbering formats are
# inconsistent enough (see extraction/supersession.py) without adding a
# second, noisier layer of guessing. A partial-relation reference with no
# detected clause is left alone entirely (status unchanged) rather than
# guessed at, or worse, applied to the whole document.
#
# Usage: python -m extraction.resolve_supersession
from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict

from extraction.supersession import FULL_RELATIONS, INFORMATIONAL_RELATIONS, PARTIAL_RELATIONS
from storage import db as storedb

logger = logging.getLogger("extraction.resolve_supersession")


def _normalize_number(number: str | None) -> str | None:
    if not number:
        return None
    normalized = re.sub(r"[^A-Za-z0-9]", "", number).lower()
    return normalized or None


def _normalize_clause(clause: str | None) -> str | None:
    if not clause:
        return None
    normalized = clause.strip().strip("().").lower()
    return normalized or None


def _clause_matches(chunk_clause_ref: str | None, referenced_clause: str) -> bool:
    """True if `chunk_clause_ref` (e.g. "4.1") is the referenced clause
    (e.g. "4") or one of its sub-clauses. Exact match or dotted-prefix
    match only -- no fuzzy numeric comparison, since clause numbering
    conventions vary too much to guess beyond "is this the same clause or
    a child of it"."""
    chunk_norm = _normalize_clause(chunk_clause_ref)
    ref_norm = _normalize_clause(referenced_clause)
    if not chunk_norm or not ref_norm:
        return False
    return chunk_norm == ref_norm or chunk_norm.startswith(ref_norm + ".")


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
        for row in conn.execute("SELECT doc_id, title, document_number, date_parsed FROM documents")
    }
    by_normalized_number: dict[str, str] = {}
    for doc_id, row in documents.items():
        norm = _normalize_number(row["document_number"])
        if norm:
            by_normalized_number[norm] = doc_id

    chunks_by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    current_status: dict[str, str] = {}
    for row in conn.execute("SELECT chunk_id, doc_id, clause_ref, status FROM chunks"):
        chunks_by_doc[row["doc_id"]].append((row["chunk_id"], row["clause_ref"] or ""))
        current_status[row["chunk_id"]] = row["status"] or "current"

    refs = conn.execute(
        "SELECT doc_id, relation, referenced_number, referenced_clause, referenced_date_raw FROM supersession_refs"
    ).fetchall()

    stats = {
        "refs_checked": 0,
        "unmatched": 0,
        "skipped_date_order": 0,
        "skipped_unscoped_partial": 0,
        "informational_skipped": 0,
        "chunks_superseded": 0,
        "chunks_amended": 0,
        "chunks_ambiguous": 0,
        "documents_fully_superseded": 0,
    }

    # chunk_id -> list of (status, note, superseder_date, superseder_id) --
    # multiple refs can propose a status for the same chunk (e.g. two
    # different later documents both claiming to supersede the same old
    # one); collected first, then reconciled in one pass below so a
    # same-status collision can pick a winner (most recent date) while a
    # cross-status collision (e.g. 'superseded' vs 'amended') becomes
    # 'ambiguous' instead of one silently overwriting the other.
    proposals: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    for row in refs:
        stats["refs_checked"] += 1
        relation = row["relation"]
        if relation in INFORMATIONAL_RELATIONS:
            stats["informational_skipped"] += 1
            continue

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

        superseder_date = superseder["date_parsed"] or ""
        target_date = target["date_parsed"] or ""
        if superseder_date and target_date and superseder_date < target_date:
            # The doc claiming to affect something is dated *before* the
            # target -- almost certainly a mismatched number, not a real
            # relationship. Skip rather than record a nonsensical link.
            stats["skipped_date_order"] += 1
            continue

        summary = _format_summary(superseder["title"], superseder["document_number"], superseder["date_parsed"])
        clause = row["referenced_clause"]

        if relation in FULL_RELATIONS:
            # A full "in supersession of..." always replaces the whole
            # document -- any clause number picked up incidentally in the
            # same lookahead window (e.g. "...read with Clause 5 of the
            # General Rules") is not a scoping signal here and is ignored;
            # scoping only matters for PARTIAL_RELATIONS below.
            for chunk_id, _ in chunks_by_doc.get(target_id, []):
                proposals[chunk_id].append(("superseded", summary, superseder_date, superseder_id))
        elif relation in PARTIAL_RELATIONS and clause:
            matched_any = False
            for chunk_id, chunk_clause_ref in chunks_by_doc.get(target_id, []):
                if _clause_matches(chunk_clause_ref, clause):
                    note = f"Amended (Clause {clause}) by {summary}"
                    proposals[chunk_id].append(("amended", note, superseder_date, superseder_id))
                    matched_any = True
            if not matched_any:
                stats["skipped_unscoped_partial"] += 1
        else:
            # A partial-relation reference ("amends...", "in partial
            # modification of...") with no clause named at all -- the
            # affected scope genuinely isn't known, so the target's
            # existing status is left untouched rather than guessed.
            stats["skipped_unscoped_partial"] += 1

    # Reconcile: multiple proposals of the same status -> most recent wins
    # (the winning proposal's doc id is kept too, needed below for the
    # whole-document rollup); proposals of differing statuses for the same
    # chunk -> ambiguous, with no single "winner" id.
    final: dict[str, tuple[str, str, str | None]] = {}  # chunk_id -> (status, note, winner_doc_id)
    for chunk_id, chunk_proposals in proposals.items():
        statuses = {p[0] for p in chunk_proposals}
        if len(statuses) == 1:
            status = next(iter(statuses))
            winner = max(chunk_proposals, key=lambda p: p[2])
            final[chunk_id] = (status, winner[1], winner[3])
        else:
            notes = sorted({p[1] for p in chunk_proposals})
            final[chunk_id] = ("ambiguous", "Conflicting supersession signals: " + "; ".join(notes), None)

    for status, group in (
        ("superseded", [cid for cid, (s, _, _) in final.items() if s == "superseded"]),
        ("amended", [cid for cid, (s, _, _) in final.items() if s == "amended"]),
        ("ambiguous", [cid for cid, (s, _, _) in final.items() if s == "ambiguous"]),
    ):
        for chunk_id in group:
            storedb.set_chunk_status(conn, [chunk_id], status, final[chunk_id][1])
        stats[f"chunks_{status}"] = len(group)

    if vector_store is not None and final:
        by_status_note: dict[tuple[str, str], list[str]] = defaultdict(list)
        for chunk_id, (status, note, _) in final.items():
            by_status_note[(status, note)].append(chunk_id)
        for (status, note), chunk_ids in by_status_note.items():
            try:
                vector_store.update_metadata(chunk_ids, {"status": status, "status_note": note})
            except Exception as exc:
                logger.warning("Vector metadata update failed for %d chunks: %s", len(chunk_ids), exc)

    # Document-level rollup: only mark a document as fully superseded (for
    # the whole-document convenience fields / dashboard stat) when EVERY one
    # of its chunks ended up 'superseded' by the SAME superseding document
    # (and none are ambiguous) -- a document with even one 'amended' or
    # 'current' chunk, or chunks superseded piecemeal by different
    # documents, still has force / is too inconsistent to roll up cleanly.
    for target_id, chunk_list in chunks_by_doc.items():
        chunk_ids = [c[0] for c in chunk_list]
        if not chunk_ids:
            continue
        resolved_statuses = set()
        superseder_ids = set()
        summary_note = None
        for chunk_id in chunk_ids:
            status, note, winner_id = final.get(
                chunk_id, (current_status.get(chunk_id, "current"), None, None)
            )
            resolved_statuses.add(status)
            if status == "superseded":
                summary_note = note
                superseder_ids.add(winner_id)
        if resolved_statuses == {"superseded"} and len(superseder_ids) == 1:
            superseder_id = next(iter(superseder_ids))
            storedb.set_document_superseded(conn, target_id, superseder_id or "", summary_note or "")
            stats["documents_fully_superseded"] += 1
            logger.info("Fully superseded: %s -> %s", target_id, summary_note)

    conn.close()
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve detected supersession/amendment references into per-chunk status on indexed documents."
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
