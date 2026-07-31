# CLI entrypoint: python -m scraper.run_scraper
#
# IMPORTANT: the target site (indianrailways.gov.in) blocks non-Indian /
# datacenter IP ranges. This must be run from a normal Indian residential or
# mobile network connection, not a cloud/CI/dev-container environment.
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scraper.crawler import IRCircularScraper

DEFAULT_ROOT_URL = (
    "https://indianrailways.gov.in/railwayboard/view_section.jsp?id=0,1,304,366,555"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl the Railway Board Traffic Commercial Directorate section."
    )
    parser.add_argument("--root-url", default=DEFAULT_ROOT_URL, help="Root menu URL to start crawling from.")
    parser.add_argument("--out-dir", default="documents", help="Directory to save documents + index into.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional cap on menu pages visited.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests (~1 req/sec default).")
    parser.add_argument("--log-file", default=None, help="Optional path to also write logs to a file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser


def configure_logging(verbose: bool, log_file: str | None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose, args.log_file)
    logger = logging.getLogger("scraper.run_scraper")

    logger.warning(
        "This scraper must be run on a normal Indian network connection — "
        "indianrailways.gov.in blocks non-Indian / datacenter IPs."
    )

    scraper = IRCircularScraper(
        root_url=args.root_url,
        out_dir=args.out_dir,
        delay=args.delay,
        max_pages=args.max_pages,
    )
    records = scraper.crawl()

    ok = sum(1 for r in records if r.download_ok)
    failed = len(records) - ok
    logger.info(
        "Crawl complete. Pages visited: %d. Documents indexed: %d (ok=%d, failed=%d). Output: %s",
        scraper.pages_visited,
        len(records),
        ok,
        failed,
        Path(args.out_dir).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
