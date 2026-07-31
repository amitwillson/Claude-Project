# CLI: python ask.py "question" [--exhaustive]
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python ask.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from qa.answer import answer_question, page_label
from qa.retrieval import retrieve
from storage import db as storedb


def main(argv=None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ask a question against the indexed circulars.")
    parser.add_argument("question", help="Your question, in quotes.")
    parser.add_argument("--exhaustive", action="store_true", help="Force exhaustive retrieval mode.")
    parser.add_argument("--db-path", default="data/ir_kb.sqlite3")
    parser.add_argument("--vector-dir", default="data/chroma_db")
    parser.add_argument("--model", default=None, help="Override ANTHROPIC_MODEL env var.")
    args = parser.parse_args(argv)

    conn = storedb.connect(args.db_path)

    vector_store = None
    embedder = None
    try:
        from storage.embeddings import get_default_provider
        from storage.vectorstore import VectorStore

        embedder = get_default_provider()
        vector_store = VectorStore(args.vector_dir)
    except Exception as exc:
        print(f"[warning] semantic search unavailable ({exc}); falling back to keyword search only.", file=sys.stderr)

    chunks, mode = retrieve(
        args.question,
        conn,
        vector_store=vector_store,
        embedder=embedder,
        exhaustive=True if args.exhaustive else None,
    )

    answer = answer_question(args.question, chunks, mode=mode, model=args.model)

    print(f"\n=== Answer (mode: {answer.mode}, model: {answer.model}) ===\n")
    print(answer.text)
    print("\n=== Sources used ===")
    if not answer.citations:
        print("(none)")
    for i, c in enumerate(answer.citations, start=1):
        print(f"{i}. {c.title} — {c.date or 'unknown date'} — {c.section_path}")
        print(f"   Clause: {c.clause_ref or 'none detected'} — Page: {page_label(c.page_start, c.page_end)}")
        print(f"   Source: {c.source_url}")
        print(f"   Local PDF: {c.local_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
