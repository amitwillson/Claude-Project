# Minimal local HTTP server that mimics the view_section.jsp?id=... menu tree
# structure for testing the crawler, without touching the real (blocked) site.
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "mock_site"

ID_TO_FILE = {
    "0,1,304,366,555": "menu_root.html",
    "0,1,304,366,555,100": "menu_100.html",
    "0,1,304,366,555,200": "menu_200.html",
    "0,1,304,366,555,201": "menu_201.html",
}


class MockSiteHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silence default request logging
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if "view_section.jsp" in parsed.path:
            qs = parse_qs(parsed.query)
            menu_id = qs.get("id", [None])[0]
            filename = ID_TO_FILE.get(menu_id)
            if filename is None:
                self.send_response(404)
                self.end_headers()
                return
            content = (FIXTURES_DIR / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(content)
            return

        # Document requests: match by basename against fixtures/mock_site/docs/
        basename = Path(parsed.path).name
        doc_path = FIXTURES_DIR / "docs" / basename
        if doc_path.exists():
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.end_headers()
            self.wfile.write(doc_path.read_bytes())
            return

        self.send_response(404)
        self.end_headers()


class MockSiteServer:
    """Context-manager wrapper around a background HTTPServer thread."""

    def __init__(self):
        self.httpd = HTTPServer(("127.0.0.1", 0), MockSiteHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def root_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/railwayboard/view_section.jsp?id=0,1,304,366,555"
