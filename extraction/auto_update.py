# Automatic "check for new circulars and index them" pipeline -- run in the
# background, on dashboard launch, so the index doesn't go stale between
# manual pipeline runs. See qa/dashboard.py for the caller.
#
# Runs, in order:
#   1. scraper       -- incremental, skips already-downloaded documents.
#      Requires a normal Indian network connection (see README); if that's
#      unavailable this step fails fast and the rest of the pipeline is
#      skipped for this attempt -- the existing index is untouched either
#      way, so a failed check never makes things worse.
#   2. extraction     -- incremental, skips documents whose content hash
#      hasn't changed. Free Tesseract OCR only.
#   3. supersession resolution -- cheap; re-resolves the whole corpus every
#      time, so a newly-added circular that amends/supersedes an existing
#      one is picked up automatically.
#   4. Claude vision OCR -- PAID last resort for documents Tesseract still
#      couldn't read after step 2. Capped at MAX_VISION_OCR_PER_RUN per
#      attempt (see below) so a bad batch can't run up an unbounded bill in
#      one unattended run; any remainder is picked up on a later check.
#
# Gated to at most once per `interval_seconds` (default: once a day) via a
# small JSON state file, since step 1 alone (a full site crawl) takes real
# time even when nothing changed. See is_due()/run_if_due().
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("extraction.auto_update")

DEFAULT_STATE_PATH = "data/auto_update_state.json"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60  # once per day

# Safety cap on paid Claude-vision-OCR calls per automatic run. This is an
# unattended, no-confirmation path (unlike the manual
# extraction.claude_vision_ocr CLI), so it's deliberately bounded rather
# than processing every flagged document in one go -- a spike in flagged
# documents is recovered over several days' checks instead of one big bill.
MAX_VISION_OCR_PER_RUN = 20


@dataclass
class AutoUpdateState:
    status: str = "idle"  # idle | scraping | extracting | resolving | ocr | done | failed
    detail: str = ""
    last_success_at: Optional[float] = None
    last_attempt_at: Optional[float] = None


def _load_state(state_path: str) -> AutoUpdateState:
    path = Path(state_path)
    if not path.exists():
        return AutoUpdateState()
    try:
        return AutoUpdateState(**json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return AutoUpdateState()


def _save_state(state_path: str, state: AutoUpdateState) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state)), encoding="utf-8")


def read_status(state_path: str = DEFAULT_STATE_PATH) -> AutoUpdateState:
    """Read-only status lookup, safe to call from the UI thread while a
    background run_if_due() may be in progress on another thread -- it's
    just a file read, not a lock/coordination point."""
    return _load_state(state_path)


def is_due(state_path: str = DEFAULT_STATE_PATH, interval_seconds: float = CHECK_INTERVAL_SECONDS) -> bool:
    state = _load_state(state_path)
    if state.last_success_at is None:
        return True
    return (time.time() - state.last_success_at) >= interval_seconds


def run_if_due(
    docs_dir: str = "documents",
    db_path: str = "data/ir_kb.sqlite3",
    vector_dir: str = "data/chroma_db",
    state_path: str = DEFAULT_STATE_PATH,
    interval_seconds: float = CHECK_INTERVAL_SECONDS,
) -> AutoUpdateState:
    """Run the full pipeline if it hasn't succeeded within `interval_seconds`;
    otherwise a no-op that returns the existing state. Meant to be called
    from a background thread (see qa/dashboard.py) since the scraper step
    can take a while."""
    state = _load_state(state_path)
    if state.last_success_at is not None and (time.time() - state.last_success_at) < interval_seconds:
        return state

    state.last_attempt_at = time.time()
    state.status = "scraping"
    state.detail = "Checking indianrailways.gov.in for new circulars..."
    _save_state(state_path, state)

    try:
        from scraper.crawler import IRCircularScraper
        from scraper.run_scraper import DEFAULT_ROOT_URL

        scraper = IRCircularScraper(root_url=DEFAULT_ROOT_URL, out_dir=docs_dir)
        records = scraper.crawl()
        downloaded_ok = sum(1 for r in records if r.download_ok)

        state.status = "extracting"
        state.detail = f"Extracting text ({downloaded_ok} document(s) on record)..."
        _save_state(state_path, state)

        from extraction.run_extract import run as run_extract

        extract_stats = run_extract(docs_dir=docs_dir, db_path=db_path, vector_dir=vector_dir)

        state.status = "resolving"
        state.detail = "Resolving supersession links..."
        _save_state(state_path, state)

        from extraction.resolve_supersession import run as run_resolve_supersession

        run_resolve_supersession(db_path=db_path, vector_dir=vector_dir)

        state.status = "ocr"
        state.detail = "Running Claude vision OCR on documents Tesseract couldn't read..."
        _save_state(state_path, state)

        from extraction.claude_vision_ocr import run as run_vision_ocr

        vision_stats = run_vision_ocr(db_path=db_path, vector_dir=vector_dir, limit=MAX_VISION_OCR_PER_RUN)

        state.status = "done"
        state.last_success_at = time.time()
        state.detail = (
            f"Up to date. {extract_stats.get('processed', 0)} new/changed document(s) indexed, "
            f"{vision_stats.get('improved', 0)} recovered via Claude vision OCR."
        )
        _save_state(state_path, state)
        logger.info("Auto-update complete: %s", state.detail)
    except Exception as exc:
        # Deliberately caught broad: this runs unattended in a background
        # thread with no one watching for a traceback, and a failed check
        # (e.g. not on an Indian network right now) must never crash the
        # dashboard or corrupt the existing index -- just report it and let
        # the next scheduled check retry.
        state.status = "failed"
        state.detail = f"Auto-update failed: {exc}"
        _save_state(state_path, state)
        logger.warning("Auto-update failed: %s", exc)

    return state
