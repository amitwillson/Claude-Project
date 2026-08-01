# Indian Railways Traffic Commercial Directorate — Circular Knowledge Base & Q&A Engine

A local pipeline that scrapes circulars/policies/guidelines from the Indian
Railways Railway Board's Traffic Commercial Directorate website, extracts and
chunks their text with full source metadata, indexes them into a local
searchable knowledge base (SQLite + FTS5 keyword search + a Chroma vector
store for semantic search), and answers questions about them with Claude —
always citing the exact circular number/title and date, and refusing to
guess when nothing relevant is indexed.

## ⚠️ IMPORTANT — network requirement for Phase 1

**`indianrailways.gov.in` blocks non-Indian and datacenter IP ranges.** This
was confirmed during development: the scraper cannot reach the site from a
cloud/CI/dev-container environment (connections reset). **You must run
`python -m scraper.run_scraper` from a normal Indian residential or mobile
network connection**, not from this repo's CI, a cloud VM, or any other
datacenter IP. Every other phase (extraction, indexing, Q&A) runs fine
anywhere once `documents/` has been populated.

Because of this constraint, the scraper in this repo was built and unit-
tested against **local mock HTML fixtures** (`tests/fixtures/mock_site/`)
that mimic the real site's `view_section.jsp?id=...` menu structure and
document-listing pages — see `tests/test_crawler.py`. It has **not** been
verified against the live site from this environment. After your first real
run, inspect the resulting `documents/_index.json` / coverage report against
what you see browsing the site manually, and iterate on `scraper/crawler.py`
parsing logic (especially `_parse_links` / `_find_nearby_date`) for any
sections that return 0 or suspiciously few documents — gov.in page layouts
are known to be inconsistent across years.

## Architecture

```
scraper/      -> crawls view_section.jsp menu tree, downloads PDFs/DOC/XLS, writes documents/_index.{csv,json}
extraction/   -> PDF/Office text extraction (+ OCR fallback), paragraph-aware chunking, supersession detection
storage/      -> SQLite (documents/chunks/FTS5/supersession_refs) + local Chroma vector store + embeddings
qa/           -> retrieval (specific/exhaustive + keyword) + Claude-backed answer generation (CLI + Streamlit)
tests/        -> unit + mock-site integration tests (no network access required)
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and set ANTHROPIC_API_KEY
```

Tesseract OCR must also be installed at the OS level for the PDF OCR
fallback to work (e.g. `apt-get install tesseract-ocr poppler-utils` on
Debian/Ubuntu — `poppler-utils` is required by `pdf2image`).

## Run order

```bash
# Phase 1 — MUST run on a normal Indian network connection, not a datacenter IP.
python -m scraper.run_scraper --out-dir documents

# Phase 2 — extract text, chunk, detect supersession language, index (SQLite + FTS5 + Chroma).
python -m extraction.run_extract --docs-dir documents

# Addendum C — human-readable coverage report (discovered/downloaded/extracted/flagged).
python storage/coverage_report.py

# Optional — last-resort OCR for the small tail Tesseract still can't read
# (uses the Anthropic API, so it costs a little money -- run this only after
# the free pipeline above, and only against the documents still flagged).
python -m extraction.claude_vision_ocr --dry-run   # preview candidates, no API calls
python -m extraction.claude_vision_ocr             # actually transcribe them

# Phase 3 — ask questions.
python qa/ask.py "What is the current policy on refund of unused tickets?"
python qa/ask.py "List all circulars on parcel booking" --exhaustive   # or just phrase it that way — auto-detected

# ...or the web UI:
streamlit run qa/webapp.py        # minimal
streamlit run qa/dashboard.py     # branded dashboard: live coverage stats, styled citation cards

# Citation-accuracy validation -- run questions where YOU already know the
# correct circular, and check whether it actually got cited. Copy
# qa/example_cases.json, fill in real questions/answers, then:
python qa/validate_citations.py my_cases.json
```

Re-running `run_scraper` / `run_extract` is safe and incremental: the
scraper skips documents already on disk (resumable), and extraction skips
documents whose SHA1 hasn't changed since the last run.

## Design notes & known limitations

- **Clause/page citations**: chunks carry the PDF page(s) and clause/paragraph
  number they came from (`extraction/chunker.py` detects leading numbering
  like "3.2", "(a)", "(iii)" and carries it forward across continuation
  paragraphs), and the Q&A engine cites them, e.g. "As per Commercial
  Circular No. 45 of 2019 dated 12.06.2019, Clause 3.2, page 2...". This is
  best-effort: not every circular uses consistent numbering, and OCR'd
  scans may miss a clause marker if it wasn't recognized cleanly -- when no
  clause number is detected, only the page number is cited. True line-level
  citation isn't attempted, since extracted text doesn't reflow 1:1 with the
  visual PDF page (especially for OCR'd scans), so a literal line number
  wouldn't reliably match what's on the actual page.
- **Notification/letter number + in-document date** (`extraction/document_ref.py`):
  each document's own reference number and issue date are read directly
  from its letterhead/reference line (e.g. "No. TC-I/2020/109/1", "New Delhi,
  dated 23.03.2020"), which is more authoritative than the scraped
  listing-page metadata and takes precedence over it when detected. Also
  best-effort -- reference-number formats aren't standardized across
  directorates/years (same caveat as `extraction/supersession.py`), so a
  `None` result means "not detected," not "document has no number."
  To backfill this field into an already-indexed database without
  re-running extraction/OCR (a document's letterhead is always within its
  first chunk, already stored from the prior run), use
  `python -m extraction.backfill_document_ref` -- fast, no PDF/API access
  needed. A full `extraction/run_extract.py --force` re-run is only needed
  if you want to re-derive everything else (chunking, OCR quality) too.
- **Embeddings provider**: Anthropic's API does not currently offer a
  dedicated embeddings endpoint, so `storage/embeddings.py` defaults to a
  local, free, offline `sentence-transformers` model (`all-MiniLM-L6-v2`).
  This costs nothing and works offline, at some quality cost vs. a large
  hosted embedding model. The `EmbeddingProvider` interface is pluggable if
  you want to swap in a hosted provider later.
- **Supersession detection is heuristic** (`extraction/supersession.py`):
  regex-based matching of phrases like "in supersession of...", "in partial
  modification of...", "amends...", "in continuation of...", plus
  best-effort extraction of the referenced circular number/date from the
  surrounding text. Circular numbering formats are **not** consistent across
  years/directorates, so this will miss references in unfamiliar formats and
  should be treated as a hint for the Q&A model to surface, not ground
  truth. The Q&A system prompt explicitly instructs Claude to flag detected
  conflicts/supersessions rather than silently pick one circular as current.
- **OCR fallback**: PDFs whose native text layer looks empty or sparse
  (fewer than ~20 words/page) are re-processed with `pytesseract` +
  `pdf2image`, page-by-page (so one bad/huge page doesn't cost the whole
  document), at 300 DPI with grayscale/autocontrast preprocessing, and with
  the Hindi (`hin`) language pack if installed alongside English. Extraction
  quality (text-layer presence, word count vs. page count, OCR confidence)
  is stored per document and outliers are written to `needs_manual_review.md`
  after every extraction run. A document whose text still can't be extracted
  at all is indexed with a placeholder chunk (title/date/section + a note
  that OCR failed) rather than silently omitted, so it's still findable and
  links back to the original PDF.
- **`extraction/claude_vision_ocr.py`** (optional, costs a little API money):
  a last-resort OCR pass for whatever's still flagged after the free
  Tesseract pipeline above, using Claude's vision to transcribe each page
  image. Deliberately a separate manual step, not run automatically, since
  it's paid — see "Run order" above.
- **Retrieval modes**: "specific" (default, vector top-N) vs. "exhaustive"
  (large candidate set + FTS5 keyword union), auto-triggered by phrasing like
  "list all" / "every circular" / "summarize all" / "all rules on", or
  forced with `--exhaustive` / the Streamlit mode toggle.
- **Old binary `.doc`/`.xls` files** (pre-2007 binary Office formats) have no
  reliable pure-Python extractor in this stack and are flagged
  `needs_review` with empty text rather than silently dropped.

## Testing

```bash
pip install -r requirements.txt   # or at minimum: requests, beautifulsoup4, pytest
python -m pytest tests/ -v
```

The test suite runs entirely against local fixtures (`tests/fixtures/mock_site/`
served by a background `http.server` instance in `tests/mock_server.py`) — it
never touches the real indianrailways.gov.in site, so it works in this
sandboxed environment and in CI.
