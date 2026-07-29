# Best-effort, regex-based detector for supersession/amendment language
# ("in supersession of...", "in partial modification of...", "amends...",
# "in continuation of...") and extraction of the referenced circular
# number/date.
#
# KNOWN HARD PROBLEM: circular numbering and reference formats are NOT
# consistent across years or directorates. This module intentionally uses
# several loose patterns and returns *candidate* references rather than
# claiming certainty — downstream consumers (retrieval/answer generation)
# should treat these as hints for surfacing possible supersession, not as an
# authoritative citation graph. Flagged here and in the README as an area
# that will need ongoing manual curation as more document formats are seen.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_RELATIONS = [
    ("supersession", re.compile(r"in\s+supersession\s+of", re.IGNORECASE)),
    ("partial_modification", re.compile(r"in\s+partial\s+modification\s+of", re.IGNORECASE)),
    ("modification", re.compile(r"in\s+modification\s+of", re.IGNORECASE)),
    ("amendment", re.compile(r"\bamends?\b|\bamendment\s+to\b", re.IGNORECASE)),
    ("continuation", re.compile(r"in\s+continuation\s+of", re.IGNORECASE)),
]

# e.g. "Commercial Circular No. 45 of 2019 dated 12.06.2019"
# e.g. "Circular No.45/2019 dated 12-06-2019"
# e.g. "letter no. TC-II/2020/45 dated 3rd March 2020"
_CIRCULAR_REF_RE = re.compile(
    r"""
    (?P<label>Commercial\s+Circular|Circular|letter|Letter)\s*
    (?:No\.?|Number)?\s*
    (?P<number>[A-Za-z0-9./\-]+)
    (?:\s+of\s+(?P<year>\d{4}))?
    (?:\s*,?\s*dated\s+(?P<date>\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4}))?
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class SupersessionRef:
    relation: str  # supersession | partial_modification | modification | amendment | continuation
    referenced_label: Optional[str]  # "Commercial Circular", "Circular", "letter", ...
    referenced_number: Optional[str]
    referenced_year: Optional[str]
    referenced_date_raw: Optional[str]
    matched_sentence: str


# How far past a relation phrase ("in supersession of", ...) to look for the
# referenced circular number/date. A fixed character window sidesteps the
# ambiguity of sentence-boundary detection around abbreviations like "No."
# or "letter no." (which a naive sentence splitter would mis-split on).
_LOOKAHEAD_CHARS = 200


def find_supersession_refs(text: str) -> list[SupersessionRef]:
    """Scan text for supersession/amendment/continuation language and try to
    extract the referenced circular number/date from the text immediately
    following the relation phrase. Best-effort — see module docstring above."""
    refs: list[SupersessionRef] = []
    for relation, pattern in _RELATIONS:
        for rel_match in pattern.finditer(text):
            window_start = rel_match.start()
            window_end = min(len(text), rel_match.end() + _LOOKAHEAD_CHARS)
            window = text[window_start:window_end]

            m = _CIRCULAR_REF_RE.search(window, pos=rel_match.end() - window_start)
            snippet = window[: m.end()] if m else window.split(".")[0]

            refs.append(
                SupersessionRef(
                    relation=relation,
                    referenced_label=m.group("label") if m else None,
                    referenced_number=m.group("number") if m else None,
                    referenced_year=m.group("year") if m else None,
                    referenced_date_raw=m.group("date") if m else None,
                    matched_sentence=snippet.strip(),
                )
            )
    return refs
