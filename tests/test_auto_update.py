from extraction import auto_update


def test_is_due_when_no_state_file_exists(tmp_path):
    state_path = str(tmp_path / "state.json")
    assert auto_update.is_due(state_path) is True


def test_is_due_false_shortly_after_a_success(tmp_path):
    state_path = str(tmp_path / "state.json")
    auto_update._save_state(state_path, auto_update.AutoUpdateState(status="done", last_success_at=auto_update.time.time()))
    assert auto_update.is_due(state_path, interval_seconds=auto_update.CHECK_INTERVAL_SECONDS) is False


def test_is_due_true_after_interval_elapses(tmp_path):
    state_path = str(tmp_path / "state.json")
    stale = auto_update.time.time() - 100
    auto_update._save_state(state_path, auto_update.AutoUpdateState(status="done", last_success_at=stale))
    assert auto_update.is_due(state_path, interval_seconds=10) is True


def test_run_if_due_skips_when_not_due(tmp_path, monkeypatch):
    state_path = str(tmp_path / "state.json")
    auto_update._save_state(state_path, auto_update.AutoUpdateState(status="done", last_success_at=auto_update.time.time()))

    def _boom(*a, **k):
        raise AssertionError("pipeline should not run when not due")

    monkeypatch.setattr("scraper.crawler.IRCircularScraper", _boom)
    result = auto_update.run_if_due(state_path=state_path)
    assert result.status == "done"


def test_run_if_due_runs_full_pipeline_and_records_success(tmp_path, monkeypatch):
    state_path = str(tmp_path / "state.json")
    calls = []

    class _FakeRecord:
        download_ok = True

    class _FakeScraper:
        def __init__(self, root_url, out_dir):
            calls.append(("scrape", out_dir))

        def crawl(self):
            return [_FakeRecord()]

    monkeypatch.setattr("scraper.crawler.IRCircularScraper", _FakeScraper)
    monkeypatch.setattr("scraper.run_scraper.DEFAULT_ROOT_URL", "https://example.com")
    monkeypatch.setattr(
        "extraction.run_extract.run",
        lambda **kw: calls.append(("extract", kw)) or {"processed": 3, "skipped": 0, "flagged": 0},
    )
    monkeypatch.setattr(
        "extraction.resolve_supersession.run",
        lambda **kw: calls.append(("resolve", kw)) or {"chunks_superseded": 0},
    )
    monkeypatch.setattr(
        "extraction.claude_vision_ocr.run",
        lambda **kw: calls.append(("vision", kw)) or {"retried": 1, "improved": 1, "still_flagged": 0},
    )

    result = auto_update.run_if_due(docs_dir="documents", db_path="db.sqlite3", vector_dir="chroma", state_path=state_path)

    assert result.status == "done"
    assert result.last_success_at is not None
    assert [c[0] for c in calls] == ["scrape", "extract", "resolve", "vision"]
    # The paid OCR step must be capped, not unbounded, on an automatic run.
    assert calls[3][1]["limit"] == auto_update.MAX_VISION_OCR_PER_RUN


def test_run_if_due_records_failure_without_crashing(tmp_path, monkeypatch):
    state_path = str(tmp_path / "state.json")

    def _fail(root_url, out_dir):
        raise ConnectionError("blocked: not on an Indian network")

    monkeypatch.setattr("scraper.crawler.IRCircularScraper", _fail)
    monkeypatch.setattr("scraper.run_scraper.DEFAULT_ROOT_URL", "https://example.com")

    result = auto_update.run_if_due(state_path=state_path)
    assert result.status == "failed"
    assert "blocked" in result.detail
    assert result.last_success_at is None
