# Citation-accuracy validation harness -- the Phase 3 spec requirement to
# "test against real questions with known correct answers... to validate
# citation accuracy before treating it as reliable," made repeatable rather
# than a one-off manual check.
#
# You supply a JSON file of test cases: questions where YOU already know
# which circular should be cited. The script runs each through the real
# retrieval + Claude answering pipeline and checks whether the expected
# circular's notification/letter number (or a keyword, as a fallback) shows
# up in the citations actually returned -- calling the Anthropic API once
# per case, so this costs a small amount of real money to run, same as any
# other question.
#
# Usage:
#   python qa/validate_citations.py cases.json
#
# cases.json format: a JSON list of objects, each with:
#   "question"        (required) -- the question to ask
#   "expected_number"  (optional) -- substring to look for in any returned
#                       citation's circular_number (case-insensitive)
#   "expected_keyword" (optional) -- substring to look for in any returned
#                       citation's title (case-insensitive), used when you
#                       know the right circular but not its exact number
#   "exhaustive"        (optional) -- force exhaustive mode for this case
#   "notes"             (optional) -- free text, printed in the report only
#
# At least one of expected_number / expected_keyword should be set per case,
# or there's nothing to check against.
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from qa.answer import answer_question
from qa.retrieval import retrieve
from storage import db as storedb


@dataclass
class CaseResult:
    question: str
    passed: bool
    reason: str
    answer_text: str
    cited_numbers: list[str]
    cited_titles: list[str]


def run_case(case: dict, conn, vector_store, embedder) -> CaseResult:
    question = case["question"]
    expected_number = case.get("expected_number")
    expected_keyword = case.get("expected_keyword")
    exhaustive = case.get("exhaustive")

    chunks, mode = retrieve(question, conn, vector_store=vector_store, embedder=embedder, exhaustive=exhaustive)
    answer = answer_question(question, chunks, mode=mode)

    cited_numbers = [c.circular_number for c in answer.citations if c.circular_number]
    cited_titles = [c.title for c in answer.citations if c.title]

    if not expected_number and not expected_keyword:
        return CaseResult(
            question, passed=False, reason="no expected_number/expected_keyword given -- nothing to check",
            answer_text=answer.text, cited_numbers=cited_numbers, cited_titles=cited_titles,
        )

    if not chunks:
        return CaseResult(
            question, passed=False, reason="no chunks retrieved at all",
            answer_text=answer.text, cited_numbers=cited_numbers, cited_titles=cited_titles,
        )

    number_hit = bool(expected_number) and any(
        expected_number.lower() in (n or "").lower() for n in cited_numbers
    )
    keyword_hit = bool(expected_keyword) and any(
        expected_keyword.lower() in (t or "").lower() for t in cited_titles
    )

    if number_hit or keyword_hit:
        return CaseResult(
            question, passed=True, reason="expected circular found in citations",
            answer_text=answer.text, cited_numbers=cited_numbers, cited_titles=cited_titles,
        )

    return CaseResult(
        question, passed=False, reason="expected circular NOT found among citations",
        answer_text=answer.text, cited_numbers=cited_numbers, cited_titles=cited_titles,
    )


def run(cases_path: str, db_path: str, vector_dir: str) -> list[CaseResult]:
    load_dotenv()
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))

    conn = storedb.connect(db_path)
    vector_store = None
    embedder = None
    try:
        from storage.embeddings import get_default_provider
        from storage.vectorstore import VectorStore

        embedder = get_default_provider()
        vector_store = VectorStore(vector_dir)
    except Exception as exc:
        print(f"[warning] semantic search unavailable ({exc}); keyword search only.", file=sys.stderr)

    results = [run_case(case, conn, vector_store, embedder) for case in cases]
    conn.close()
    return results


def print_report(results: list[CaseResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    print(f"\n=== Citation accuracy: {passed}/{len(results)} passed ===\n")
    for i, r in enumerate(results, start=1):
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {i}. {r.question}")
        print(f"       reason: {r.reason}")
        print(f"       cited numbers: {r.cited_numbers}")
        print(f"       cited titles:  {r.cited_titles}")
        if not r.passed:
            print(f"       answer preview: {r.answer_text[:200]!r}")
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate Q&A citation accuracy against known-answer questions.")
    parser.add_argument("cases", help="Path to a JSON file of test cases (see module docstring for format).")
    parser.add_argument("--db-path", default="data/ir_kb.sqlite3")
    parser.add_argument("--vector-dir", default="data/chroma_db")
    args = parser.parse_args(argv)

    results = run(args.cases, args.db_path, args.vector_dir)
    print_report(results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
