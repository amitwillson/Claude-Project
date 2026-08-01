# Addendum C: coverage report. Reads the scraper's documents index plus the
# SQLite extraction state and writes a human-readable coverage_report.md.
# Regenerable/incremental: safe to re-run any time after indexing.
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python storage/coverage_report.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.models import load_index
from storage import db as storedb

LOW_COUNT_THRESHOLD = 2  # sections/years with fewer than this many docs are flagged


def build_report(docs_dir: str | Path, db_path: str | Path, out_path: str | Path) -> str:
    docs_dir = Path(docs_dir)
    records = load_index(docs_dir)

    discovered = len(records)
    downloaded = sum(1 for r in records if r.download_ok)

    extracted = 0
    flagged = 0
    with_document_number = 0
    per_section_extracted: Counter = Counter()
    per_section_discovered: Counter = Counter()
    for r in records:
        per_section_discovered[r.section_path] += 1

    db_path = Path(db_path)
    if db_path.exists():
        # storage.db.connect() (not a raw sqlite3.connect()) so schema
        # migrations always run first -- otherwise a database created by an
        # older version of this project would be missing newer columns
        # (e.g. document_number) and this query would crash.
        conn = storedb.connect(db_path)
        rows = conn.execute(
            "SELECT doc_id, section_path, needs_review, word_count, document_number FROM documents"
        ).fetchall()
        extracted = len(rows)
        flagged = sum(1 for row in rows if row["needs_review"])
        with_document_number = sum(1 for row in rows if row["document_number"])
        for row in rows:
            per_section_extracted[row["section_path"]] += 1
        conn.close()

    lines = []
    lines.append("# Coverage Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Documents discovered: **{discovered}**")
    lines.append(f"- Documents downloaded: **{downloaded}**")
    lines.append(f"- Documents extracted/indexed: **{extracted}**")
    lines.append(f"- Documents flagged for manual review: **{flagged}**")
    if extracted:
        pct = round(100 * with_document_number / extracted, 1)
        lines.append(
            f"- Documents with a detected notification/letter number: "
            f"**{with_document_number}** ({pct}%)"
        )
    lines.append("")

    lines.append("## Per-section breakdown")
    lines.append("")
    lines.append("| Section | Discovered | Extracted | Flag |")
    lines.append("| --- | ---: | ---: | --- |")
    all_sections = sorted(set(per_section_discovered) | set(per_section_extracted))
    for section in all_sections:
        d = per_section_discovered.get(section, 0)
        e = per_section_extracted.get(section, 0)
        flag = " LOW COUNT" if d <= LOW_COUNT_THRESHOLD else ""
        lines.append(f"| {section or '(root)'} | {d} | {e} | {flag} |")
    lines.append("")

    if discovered == 0:
        lines.append(
            "> No documents discovered yet — run `python -m scraper.run_scraper` first "
            "(must be run on a normal Indian network connection; see README)."
        )
        lines.append("")

    report = "\n".join(lines)
    out_path = Path(out_path)
    out_path.write_text(report, encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the indexing coverage report.")
    parser.add_argument("--docs-dir", default="documents")
    parser.add_argument("--db-path", default="data/ir_kb.sqlite3")
    parser.add_argument("--out", default="coverage_report.md")
    args = parser.parse_args(argv)

    report = build_report(args.docs_dir, args.db_path, args.out)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
