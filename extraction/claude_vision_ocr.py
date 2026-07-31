# Last-resort OCR fallback using Claude's vision, for the small tail of
# documents Tesseract genuinely can't read even with the higher-DPI +
# preprocessing pass (see pdf_extract.py). Deliberately scoped to only the
# documents already flagged needs_review=1 in the database, since this
# calls the paid Anthropic API per page -- run it after the free Tesseract
# pipeline has already done everything it can, not as a first resort.
#
# Usage: python -m extraction.claude_vision_ocr [--limit N] [--dry-run]
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from extraction.chunker import chunk_text
from extraction.document_ref import detect_document_ref
from extraction.pdf_extract import ExtractionResult, _OCR_DPI, flag_extraction_outliers
from extraction.run_extract import _write_manual_review_report
from extraction.supersession import find_supersession_refs
from storage import db as storedb

logger = logging.getLogger("extraction.claude_vision_ocr")

DEFAULT_MODEL = "claude-sonnet-5"

# A document needing vision OCR beyond this many pages would be expensive
# and slow to transcribe one page-image-per-API-call; flag it for genuine
# manual review instead of silently spending a lot on it.
MAX_PAGES_PER_DOC = 30

TRANSCRIBE_PROMPT = (
    "Transcribe ALL readable text from this scanned document page verbatim, "
    "preserving the reading order and table/list structure as plain text. "
    "The text may be in English, Hindi (Devanagari script), or a mix of both -- "
    "transcribe exactly what is written, do not translate. If a word or section "
    "is genuinely illegible, write [illegible] in its place rather than guessing. "
    "Output ONLY the transcribed text, with no commentary, preamble, or markdown."
)


def _render_page_png_base64(pdf_bytes: bytes, page_num: int) -> Optional[str]:
    from pdf2image import convert_from_bytes
    from PIL import Image

    # Some circulars are nested deep enough that the full path exceeds
    # Windows' classic 260-character MAX_PATH limit -- pdftoppm.exe (a
    # native subprocess) can't open such paths even though Python can, so
    # we read the bytes ourselves and pipe them in rather than passing a
    # file path. Also disable PIL's decompression-bomb guard: it's meant for
    # untrusted image sources, but these all come from PDFs we ourselves
    # downloaded from the Railway Board site, and oversized tariff-table
    # pages at 300 DPI can otherwise hit it as a hard error, not just a
    # warning.
    Image.MAX_IMAGE_PIXELS = None

    images = convert_from_bytes(
        pdf_bytes, first_page=page_num, last_page=page_num, dpi=_OCR_DPI, timeout=120
    )
    if not images:
        return None
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def transcribe_document(path: Path, page_count: int, model: str, client) -> list[str]:
    """Render each page and ask Claude to transcribe it. A page that fails
    to render or gets an API error is skipped (stays blank) rather than
    aborting the whole document. Returns per-page text so chunking can still
    tag clause/page citations, same as the Tesseract path."""
    n_pages = min(max(page_count, 1), MAX_PAGES_PER_DOC)
    pdf_bytes = path.read_bytes()
    all_text = []
    for page_num in range(1, n_pages + 1):
        try:
            img_b64 = _render_page_png_base64(pdf_bytes, page_num)
            if img_b64 is None:
                all_text.append("")
                continue
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                            },
                            {"type": "text", "text": TRANSCRIBE_PROMPT},
                        ],
                    }
                ],
            )
            text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            all_text.append("\n".join(text_parts).strip())
        except Exception as exc:
            logger.warning("Vision OCR failed for %s page %d: %s", path, page_num, exc)
            all_text.append("")
    return all_text


def run(
    db_path: str,
    vector_dir: str,
    limit: Optional[int] = None,
    dry_run: bool = False,
    model: Optional[str] = None,
) -> dict:
    load_dotenv()
    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)

    conn = storedb.connect(db_path)
    rows = conn.execute(
        "SELECT doc_id, title, section_path, date_raw, date_parsed, source_url, "
        "local_path, sha1, doc_type, page_count FROM documents "
        "WHERE needs_review = 1 AND doc_type = 'pdf' ORDER BY doc_id"
    ).fetchall()
    if limit:
        rows = list(rows)[:limit]

    logger.info("Found %d flagged PDF(s) to retry with Claude vision.", len(rows))
    if dry_run:
        for row in rows:
            print(f"{row['local_path']}  (pages={row['page_count']})")
        conn.close()
        return {"candidates": len(rows)}

    import anthropic

    client = anthropic.Anthropic()

    vector_store = None
    embedder = None
    try:
        from storage.embeddings import get_default_provider
        from storage.vectorstore import VectorStore

        embedder = get_default_provider()
        vector_store = VectorStore(vector_dir)
    except Exception as exc:
        logger.warning("Semantic index unavailable (%s) -- keyword search only.", exc)

    improved = 0
    still_flagged = 0
    for row in rows:
        local_path = Path(row["local_path"]) if row["local_path"] else None
        if not local_path or not local_path.exists():
            logger.warning("SKIP missing file: %s", row["local_path"])
            continue

        page_count = row["page_count"] or 1
        pages = transcribe_document(local_path, page_count, model, client)
        text = "\n\n".join(pages)
        word_count = len(text.split())

        result = ExtractionResult(
            text=text, page_count=page_count, word_count=word_count,
            has_text_layer=False, ocr_used=True, ocr_confidence=None,
        )
        needs_review = flag_extraction_outliers(result)
        if needs_review:
            still_flagged += 1
        else:
            improved += 1

        # Same preference as run_extract.py: the document's own letterhead
        # is more authoritative than the scraped listing-page date.
        ref = detect_document_ref(text)
        date_raw = ref.date_raw or row["date_raw"]
        date_parsed = ref.date_parsed or row["date_parsed"]

        doc_row = {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "section_path": row["section_path"],
            "date_raw": date_raw,
            "date_parsed": date_parsed,
            "source_url": row["source_url"],
            "local_path": row["local_path"],
            "sha1": row["sha1"],
            "doc_type": row["doc_type"],
            "has_text_layer": 0,
            "ocr_used": 1,
            "ocr_confidence": None,
            "word_count": word_count,
            "page_count": page_count,
            "extraction_error": "Claude vision OCR: still low confidence" if needs_review else None,
            "needs_review": int(needs_review),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "document_number": ref.number,
        }
        storedb.upsert_document(conn, doc_row)

        storedb.delete_chunks_for_doc(conn, row["doc_id"])
        indexable_text = pages if text.strip() else (
            f"[Claude vision OCR could not extract readable text from this document. "
            f"Title: {row['title']}. Section: {row['section_path']}. "
            f"Date: {date_parsed or date_raw or 'unknown'}. "
            f"Refer to the original PDF ({row['local_path']}) for its content.]"
        )
        chunks = chunk_text(
            text=indexable_text,
            doc_id=row["doc_id"],
            section_path=row["section_path"],
            title=row["title"],
            date=date_parsed or date_raw,
            source_url=row["source_url"],
            local_path=row["local_path"],
            doc_type=row["doc_type"],
            circular_number=ref.number,
        )
        storedb.insert_chunks(conn, chunks)

        if text.strip():
            refs = find_supersession_refs(text)
            storedb.insert_supersession_refs(conn, row["doc_id"], refs)

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
            "Vision OCR %s (%d words, needs_review=%s)",
            row["local_path"], word_count, needs_review,
        )

    # Refresh needs_manual_review.md from the DB's current (post-vision) state,
    # not just this run's own results -- it must stay a complete, accurate list.
    remaining = conn.execute(
        "SELECT title, section_path, local_path, extraction_error FROM documents WHERE needs_review = 1"
    ).fetchall()
    entries = [
        f"- **{r['title']}** ({r['section_path']}) — `{r['local_path']}` — "
        f"{r['extraction_error'] or 'low word count / possible extraction loss'}"
        for r in remaining
    ]
    _write_manual_review_report(entries)

    conn.close()
    return {"retried": len(rows), "improved": improved, "still_flagged": still_flagged}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Last-resort Claude-vision OCR for documents Tesseract couldn't read."
    )
    parser.add_argument("--db-path", default="data/ir_kb.sqlite3")
    parser.add_argument("--vector-dir", default="data/chroma_db")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N flagged documents.")
    parser.add_argument("--dry-run", action="store_true", help="List candidates without calling the API.")
    parser.add_argument("--model", default=None, help="Override ANTHROPIC_MODEL env var.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stats = run(args.db_path, args.vector_dir, limit=args.limit, dry_run=args.dry_run, model=args.model)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
