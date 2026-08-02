# Optional general-web fallback for questions that aren't about indexed
# circulars at all (e.g. "what is RAC in Indian Railways ticketing?" as a
# basic definitional question) -- NOT a substitute for the circular-grounded
# answer path in qa/answer.py, which must keep citing only indexed documents
# and saying "No matching circular found" rather than guess (see
# qa/answer.py SYSTEM_PROMPT rule 1). This module is only ever consulted
# AFTER that path has already come back empty, and its output is always
# rendered separately and labeled as an unverified general web result -- see
# qa/dashboard.py.
#
# Uses the DuckDuckGo Instant Answer API: free, keyless, no ToS-violating
# scraping (unlike hitting Google's HTML search directly, which risks the
# same kind of IP blocking already documented for indianrailways.gov.in).
# Trade-off: it only returns infobox-style abstracts/definitions, not full
# search results -- fine for "basic answers", not a real search engine.
from __future__ import annotations

import urllib.parse
import urllib.request
import json
from dataclasses import dataclass
from typing import Optional

DUCKDUCKGO_ENDPOINT = "https://api.duckduckgo.com/"
TIMEOUT_SECONDS = 6


@dataclass
class WebAnswer:
    text: str
    source_url: str
    source_name: str


def search_basic_answer(query: str) -> Optional[WebAnswer]:
    """Best-effort lookup of a basic factual answer from the free DuckDuckGo
    Instant Answer API. Returns None on any failure (network unavailable,
    no instant answer for this query, malformed response, etc.) -- this is
    a nice-to-have fallback, never something the caller should treat as
    required to succeed."""
    params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
    url = f"{DUCKDUCKGO_ENDPOINT}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        return WebAnswer(
            text=abstract,
            source_url=data.get("AbstractURL", "") or "",
            source_name=data.get("AbstractSource", "") or "the web",
        )

    for topic in data.get("RelatedTopics", []) or []:
        text = (topic.get("Text") or "").strip()
        if text:
            return WebAnswer(text=text, source_url=topic.get("FirstURL", "") or "", source_name="the web")

    return None
