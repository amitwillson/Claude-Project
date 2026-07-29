# Integration test: run IRCircularScraper against the local mock site fixture
# server and assert menu recursion terminates, docs are found, index files
# are written correctly, dedup by id works, and re-runs are resumable.
from __future__ import annotations

import json

from scraper.crawler import IRCircularScraper
from tests.mock_server import MockSiteServer


def test_crawl_discovers_all_documents_and_terminates(tmp_path):
    with MockSiteServer() as server:
        scraper = IRCircularScraper(root_url=server.root_url, out_dir=tmp_path, delay=0)
        records = scraper.crawl()

    # 4 documents live under menu_100.html.
    assert len(records) == 4
    assert all(r.download_ok for r in records)

    # Menu recursion terminated: root, id=100 (visited once despite self-link),
    # id=200, id=201 — 4 unique menu nodes, not infinite.
    assert scraper.visited_menu_ids == {"0,1,304,366,555", "100", "200", "201"}


def test_crawl_dedup_by_id_visits_each_menu_once(tmp_path):
    with MockSiteServer() as server:
        scraper = IRCircularScraper(root_url=server.root_url, out_dir=tmp_path, delay=0)
        scraper.crawl()

    # menu_100.html contains a self-link back to id=100; it must not be
    # re-fetched (pages_visited counts each unique menu id once).
    assert scraper.pages_visited == 4


def test_index_files_written_correctly(tmp_path):
    with MockSiteServer() as server:
        scraper = IRCircularScraper(root_url=server.root_url, out_dir=tmp_path, delay=0)
        scraper.crawl()

    index_json = tmp_path / "_index.json"
    index_csv = tmp_path / "_index.csv"
    assert index_json.exists()
    assert index_csv.exists()

    data = json.loads(index_json.read_text())
    assert len(data) == 4
    titles = {d["title"] for d in data}
    assert any("Refund Rules" in t for t in titles)
    assert any("supersession" in t for t in titles)

    # Date parsing across the three formats present in the fixture.
    by_title = {d["title"]: d for d in data}
    dotted = next(d for t, d in by_title.items() if "Refund Rules" in t)
    assert dotted["date_parsed"] == "2020-06-12"
    dashed = next(d for t, d in by_title.items() if "Parcel Booking" in t)
    assert dashed["date_parsed"] == "2019-08-15"
    textual = next(d for t, d in by_title.items() if "Luggage Rules" in t)
    assert textual["date_parsed"] == "2021-03-03"

    # sha1 + local_path populated for every downloaded doc.
    for d in data:
        assert d["sha1"]
        assert d["local_path"]
        assert d["doc_type"] == "pdf"


def test_downloaded_files_exist_on_disk_with_correct_hash(tmp_path):
    import hashlib

    with MockSiteServer() as server:
        scraper = IRCircularScraper(root_url=server.root_url, out_dir=tmp_path, delay=0)
        records = scraper.crawl()

    for r in records:
        from pathlib import Path

        p = Path(r.local_path)
        assert p.exists()
        assert hashlib.sha1(p.read_bytes()).hexdigest() == r.sha1


def test_resumability_second_run_downloads_zero_new_files(tmp_path):
    with MockSiteServer() as server:
        scraper1 = IRCircularScraper(root_url=server.root_url, out_dir=tmp_path, delay=0)
        records1 = scraper1.crawl()
        assert len(records1) == 4

        # Second run against a fresh scraper instance pointed at the same
        # out_dir: every doc URL is already in the index / already on disk,
        # so no new downloads should happen and the index should stay at 4.
        scraper2 = IRCircularScraper(root_url=server.root_url, out_dir=tmp_path, delay=0)
        records2 = scraper2.crawl()

    assert len(records2) == 4
    # Every record on the second run should have been resolved from the
    # existing on-disk file / index rather than freshly fetched.
    assert all(r.download_ok for r in records2)
