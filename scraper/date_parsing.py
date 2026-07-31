# Robust date parser for the varied date formats seen on Railway Board circular
# listing pages: dd.mm.yyyy, dd-mm-yyyy, dd/mm/yyyy, and "DD Mon YYYY" (with
# optional ordinal suffixes / extra whitespace / abbreviated or full month names).
from __future__ import annotations

import re
from datetime import date
from typing import Optional

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# dd.mm.yyyy / dd-mm-yyyy / dd/mm/yyyy (also accepts 2-digit years)
_NUMERIC_RE = re.compile(
    r"\b(?P<d>\d{1,2})[.\-/](?P<m>\d{1,2})[.\-/](?P<y>\d{2,4})\b"
)

# "12 June 2019", "12th Jun 2019", "12 Jun, 2019"
_TEXT_RE = re.compile(
    r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>[A-Za-z]{3,9})\.?,?\s+(?P<y>\d{4})\b"
)


def _normalize_year(y: int) -> int:
    if y < 100:
        # Heuristic: 2-digit years on Indian govt circulars are 1900s/2000s.
        return 2000 + y if y <= 69 else 1900 + y
    return y


def parse_date(text: Optional[str]) -> Optional[str]:
    """Best-effort parse of a date string into ISO 8601 (YYYY-MM-DD).

    Returns None if nothing recognizable is found. Tries numeric formats
    first (dd.mm.yyyy / dd-mm-yyyy / dd/mm/yyyy), then "DD Mon YYYY" text form.
    """
    if not text:
        return None
    text = text.strip()
    if not text:
        return None

    m = _NUMERIC_RE.search(text)
    if m:
        d, mo, y = int(m.group("d")), int(m.group("m")), _normalize_year(int(m.group("y")))
        parsed = _try_build(y, mo, d)
        if parsed:
            return parsed

    m = _TEXT_RE.search(text)
    if m:
        mon_name = m.group("mon").lower()
        mo = _MONTHS.get(mon_name)
        if mo:
            d, y = int(m.group("d")), int(m.group("y"))
            parsed = _try_build(y, mo, d)
            if parsed:
                return parsed

    return None


def _try_build(y: int, mo: int, d: int) -> Optional[str]:
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None
