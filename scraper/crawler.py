# IRCircularScraper: BFS crawl of the Railway Board view_section.jsp?id=... menu
# tree, downloading linked circular documents (PDF/DOC/DOCX/XLS/XLSX) with
# metadata capture, resumability, and polite rate limiting.
#
# Fetching is isolated behind fetch_page()/fetch_binary() so a Playwright-based
# fallback could later be swapped in for any menu node whose document listing
# turns out to require JS rendering (a known risk on gov.in sites) — the rest
# of the crawl/parse logic does not need to change.
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scraper.date_parsing import parse_date
from scraper.models import DocumentRecord, load_index, write_index

logger = logging.getLogger("scraper.crawler")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

DOC_EXTENSIONS = (".pdf", ".doc", ".docx", ".xls", ".xlsx")


@dataclass
class MenuNode:
    id: str
    url: str
    breadcrumb: Optional[str]  # full path up to and including this node's own label; None only for the root, meaning "derive from this page's own title once fetched"


def build_session(user_agent: str = USER_AGENT, total_retries: int = 5) -> requests.Session:
    """A requests.Session configured with retry/backoff and a realistic UA."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    retry = Retry(
        total=total_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTPS_ONLY_HOSTS = {"indianrailways.gov.in", "www.indianrailways.gov.in"}


def normalize_scheme(url: str) -> str:
    """Force http:// links to https:// on hosts known to no longer accept
    plain-http connections. Many old circular links on this site are
    hardcoded http://, but the server refuses connections on port 80, which
    otherwise burns a full 5-retry exponential backoff (~30s+) per dead link
    for no reason — https works. Scoped to the real site's host only, so it
    doesn't interfere with local/test HTTP servers."""
    parsed = urlparse(url)
    if parsed.scheme == "http" and parsed.hostname in HTTPS_ONLY_HOSTS:
        return parsed._replace(scheme="https").geturl()
    return url


def extract_menu_id(url: str) -> Optional[str]:
    """Extract the id= query param from a view_section.jsp URL."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    ids = qs.get("id")
    if not ids:
        return None
    return ids[0]


def is_document_link(href: str) -> bool:
    path = urlparse(href).path.lower()
    return any(path.endswith(ext) for ext in DOC_EXTENSIONS)


def doc_type_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in DOC_EXTENSIONS:
        if path.endswith(ext):
            return ext.lstrip(".")
    return "unknown"


class IRCircularScraper:
    """BFS crawler over the recursive view_section.jsp?id=... menu tree.

    Well-behaved: ~1 req/sec, retries with backoff, realistic UA, per-page
    logging, resumable (skips already-downloaded files by checking the
    existing index + documents/ dir).
    """

    def __init__(
        self,
        root_url: str,
        out_dir: str | Path,
        delay: float = 1.0,
        max_pages: Optional[int] = None,
        session: Optional[requests.Session] = None,
    ):
        self.root_url = root_url
        self.root_id = extract_menu_id(root_url) or "root"
        self.out_dir = Path(out_dir)
        self.docs_dir = self.out_dir
        self.delay = delay
        self.max_pages = max_pages
        self.session = session or build_session()
        # Document downloads are numerous and many old links are dead
        # (http:// on a port that now refuses connections) — fail those
        # fast instead of burning a full 5x backoff per link.
        self.binary_session = session or build_session(total_retries=1)

        self.visited_menu_ids: set[str] = set()
        self.seen_doc_urls: set[str] = set()
        self.records: list[DocumentRecord] = load_index(self.out_dir)
        for r in self.records:
            self.seen_doc_urls.add(r.source_url)

        self._last_request_time: float = 0.0
        self.pages_visited = 0

    # -- fetch layer (isolated so a JS-rendering fallback could be added) --

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch an HTML page. Returns None on failure (already retried)."""
        self._throttle()
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            logger.warning("Failed to fetch page %s: %s", url, exc)
            return None

    def fetch_binary(self, url: str) -> Optional[tuple[bytes, int]]:
        """Fetch a binary document. Returns (content, status_code) or None."""
        url = normalize_scheme(url)
        self._throttle()
        try:
            resp = self.binary_session.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content, resp.status_code
        except requests.RequestException as exc:
            logger.warning("Failed to fetch document %s: %s", url, exc)
            return None

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = self.delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    # -- crawl --

    # Flush the index to disk this often (in documents handled), so a crash,
    # closed terminal, or sleeping PC during a many-hour crawl never loses
    # everything back to zero — only the docs handled since the last flush.
    INDEX_FLUSH_EVERY = 25

    def crawl(self) -> list[DocumentRecord]:
        root_id = extract_menu_id(self.root_url) or "root"
        # breadcrumb=None is a sentinel meaning "not yet known — compute
        # from this page's own title/h1 once fetched" (only true for root;
        # every other node's breadcrumb is fixed at enqueue time below).
        queue: deque[MenuNode] = deque([MenuNode(id=root_id, url=self.root_url, breadcrumb=None)])
        docs_since_flush = 0

        try:
            while queue:
                if self.max_pages is not None and self.pages_visited >= self.max_pages:
                    logger.info("Reached max_pages=%s, stopping crawl.", self.max_pages)
                    break

                node = queue.popleft()
                if node.id in self.visited_menu_ids:
                    continue
                self.visited_menu_ids.add(node.id)

                html = self.fetch_page(node.url)
                self.pages_visited += 1
                if html is None:
                    logger.info("PAGE FAIL id=%s url=%s", node.id, node.url)
                    continue

                soup = BeautifulSoup(html, "html.parser")
                # Only the root's breadcrumb comes from the page's own
                # title/h1 — every deeper node's breadcrumb was already
                # fixed when it was enqueued, from the *parent* page's link
                # text. gov.in pages typically share one generic <title>
                # site-wide, so reading each child page's own title would
                # collapse every section into the same repeated label.
                breadcrumb = node.breadcrumb if node.breadcrumb is not None else self._page_title(soup, node)

                sub_menus, docs = self._parse_links(soup, node.url)
                logger.info(
                    "PAGE OK id=%s url=%s links=%d docs=%d",
                    node.id,
                    node.url,
                    len(sub_menus),
                    len(docs),
                )

                for doc_url, doc_title, date_text in docs:
                    self._handle_document(doc_url, doc_title, date_text, breadcrumb, node.id)
                    docs_since_flush += 1
                    if docs_since_flush >= self.INDEX_FLUSH_EVERY:
                        write_index(self.records, self.out_dir)
                        docs_since_flush = 0

                for sub_id, sub_url, sub_label in sub_menus:
                    if sub_id in self.visited_menu_ids:
                        continue
                    if not self._in_scope(sub_id):
                        logger.info("SKIP out-of-scope menu id=%s url=%s", sub_id, sub_url)
                        continue
                    label = sub_label or f"section-{sub_id}"
                    child_breadcrumb = f"{breadcrumb} > {label}".strip(" >")
                    queue.append(MenuNode(id=sub_id, url=sub_url, breadcrumb=child_breadcrumb))
        finally:
            # Always persist whatever we have, even on KeyboardInterrupt or
            # an unexpected exception mid-crawl.
            write_index(self.records, self.out_dir)

        return self.records

    def _in_scope(self, menu_id: str) -> bool:
        """Menu ids are hierarchical, comma-separated node paths (e.g.
        "0,1,304,366,555,999" is a child of "0,1,304,366,555"). The site's
        nav menu is shared across every directorate, so an unscoped BFS
        wanders into completely unrelated sections (Personnel, Vacancy
        Circulars, etc.) — restrict the crawl to the root section's own
        subtree."""
        if menu_id == self.root_id:
            return True
        return menu_id.startswith(self.root_id + ",")

    def _page_title(self, soup: BeautifulSoup, node: MenuNode) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find(["h1", "h2"])
        if h1:
            return h1.get_text(strip=True)
        return f"section-{node.id}"

    def _parse_links(
        self, soup: BeautifulSoup, page_url: str
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, Optional[str]]]]:
        """Return (sub_menu_links, document_links).

        sub_menu_links: list of (menu_id, absolute_url, link_text) — link_text
        is this page's own label for that child section (used to build the
        child's breadcrumb, since gov.in child pages don't reliably have
        their own distinct <title>).
        document_links: list of (absolute_url, title, nearby_date_text)
        """
        sub_menus: list[tuple[str, str, str]] = []
        docs: list[tuple[str, str, Optional[str]]] = []

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("javascript:") or href.startswith("#"):
                continue
            abs_url = urljoin(page_url, href)

            if is_document_link(abs_url):
                title = a.get_text(strip=True) or Path(urlparse(abs_url).path).name
                date_text = self._find_nearby_date(a)
                docs.append((abs_url, title, date_text))
                continue

            if "view_section.jsp" in abs_url:
                menu_id = extract_menu_id(abs_url)
                if menu_id:
                    link_text = a.get_text(strip=True)
                    sub_menus.append((menu_id, abs_url, link_text))

        return sub_menus, docs

    @staticmethod
    def _find_nearby_date(anchor) -> Optional[str]:
        """Look for a date-like string near the link (same row/parent, or
        immediately preceding/following text) — gov.in listing pages usually
        print the date right next to the link, in varying markup."""
        # Same table row is the most common pattern.
        row = anchor.find_parent("tr")
        if row:
            text = row.get_text(" ", strip=True)
            m = re.search(
                r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|"
                r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4}",
                text,
            )
            if m:
                return m.group(0)

        # Fall back to the parent element's text.
        parent = anchor.find_parent(["li", "p", "div", "td"])
        if parent:
            text = parent.get_text(" ", strip=True)
            m = re.search(
                r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|"
                r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4}",
                text,
            )
            if m:
                return m.group(0)

        # Fall back to the anchor's own text.
        text = anchor.get_text(" ", strip=True)
        m = re.search(
            r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|"
            r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4}",
            text,
        )
        return m.group(0) if m else None

    def _handle_document(
        self,
        doc_url: str,
        title: str,
        date_text: Optional[str],
        section_path: str,
        menu_id: str,
    ) -> None:
        if doc_url in self.seen_doc_urls:
            return  # already indexed from a previous crawl or earlier in this run
        self.seen_doc_urls.add(doc_url)

        doc_id = hashlib.sha1(doc_url.encode("utf-8")).hexdigest()
        doc_type = doc_type_from_url(doc_url)
        local_dir = self.docs_dir / self._safe_path(section_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        filename = self._safe_filename(title, doc_id, doc_type)
        local_path = local_dir / filename

        record = DocumentRecord(
            doc_id=doc_id,
            section_path=section_path,
            title=title,
            date_raw=date_text,
            date_parsed=parse_date(date_text),
            source_url=doc_url,
            local_path=str(local_path),
            doc_type=doc_type,
            menu_id=menu_id,
        )

        if local_path.exists():
            # Resumable: already downloaded in a prior run.
            record.download_ok = True
            record.sha1 = self._sha1_file(local_path)
            logger.info("SKIP (already downloaded) %s", doc_url)
        else:
            result = self.fetch_binary(doc_url)
            if result is None:
                record.download_ok = False
                record.error = "download failed"
                logger.warning("DOC FAIL %s", doc_url)
            else:
                content, status = result
                local_path.write_bytes(content)
                record.download_ok = True
                record.http_status = status
                record.sha1 = hashlib.sha1(content).hexdigest()
                logger.info("DOC OK %s -> %s", doc_url, local_path)

        self.records.append(record)

    @staticmethod
    def _sha1_file(path: Path) -> str:
        h = hashlib.sha1()
        h.update(path.read_bytes())
        return h.hexdigest()

    @staticmethod
    def _safe_path(section_path: str) -> str:
        parts = [p.strip() for p in section_path.split(">") if p.strip()]
        safe_parts = [re.sub(r'[<>:"/\\|?*]', "_", p)[:80] for p in parts]
        return str(Path(*safe_parts)) if safe_parts else "misc"

    @staticmethod
    def _safe_filename(title: str, doc_id: str, doc_type: str) -> str:
        safe_title = re.sub(r'[<>:"/\\|?*]', "_", title).strip()[:100] or "document"
        return f"{safe_title}_{doc_id[:8]}.{doc_type}"
