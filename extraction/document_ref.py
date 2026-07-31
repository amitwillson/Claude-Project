# Best-effort detector for a document's OWN notification/letter/circular
# reference number and issue date, read directly from the PDF's own text
# (the letterhead / reference line near the top of page 1) rather than
# relying solely on the scraped listing-page metadata -- which can be
# missing, truncated, or occasionally not quite matched to the right link.
# When something is detected here it takes precedence over the scraped
# date; the scraped date remains the fallback when nothing is found in the
# document text itself.
#
# KNOWN HARD PROBLEM (same caveat as extraction/supersession.py): reference
# number formats are not standardized across directorates/years. This is
# best-effort enrichment, not authoritative ground truth -- treat a None
# result as "not detected", not "document has no number".
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from scraper.date_parsing import parse_date

# Letterhead/reference lines are almost always within the first ~1500
# characters of page 1 text; scanning further risks matching an unrelated
# number mentioned later in the body (e.g. a referenced circular, a rate,
# a case number).
_HEAD_CHARS = 1500

# e.g. "No. TC-I/2020/109/1", "F.No. TCR/1078/2019/2", "Ref. No. 45/2019",
# "No.TCR-1078/2019/2-Part(1)". Requires at least one digit in the captured
# token so common phrases like "No. of days" don't false-positive.
_NUMBER_RE = re.compile(
    r"""
    (?:F\.?\s*)?
    (?:No\.?|Ref\.?\s*No\.?|Number)\s*[:.]?\s*
    (?P<number>(?=[A-Za-z0-9/_.\-()]*\d)[A-Za-z0-9][A-Za-z0-9/_.\-()]{2,40})
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Government letterhead date line: "New Delhi, dated 23.03.2020",
# "नई दिल्ली, दिनांक 23.03.2020", "Dated: 12th June 2019".
_DATED_LINE_RE = re.compile(
    r"(?:dated|dinank|दिनांक)\s*[:.,]?\s*"
    r"(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4})",
    re.IGNORECASE,
)


@dataclass
class DocumentRef:
    number: Optional[str]
    date_raw: Optional[str]
    date_parsed: Optional[str]  # ISO 8601 (YYYY-MM-DD) if parseable


def detect_document_ref(text: str) -> DocumentRef:
    """Best-effort extraction of a document's own reference number and
    issue date from the start of its extracted text."""
    head = text[:_HEAD_CHARS] if text else ""

    number = None
    m = _NUMBER_RE.search(head)
    if m:
        number = m.group("number").strip().rstrip(".,;:")

    date_raw = None
    date_parsed = None
    m = _DATED_LINE_RE.search(head)
    if m:
        date_raw = m.group(1)
        date_parsed = parse_date(date_raw)

    return DocumentRef(number=number, date_raw=date_raw, date_parsed=date_parsed)
