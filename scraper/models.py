# Data model for a single discovered/downloaded document and its metadata.
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class DocumentRecord:
    """One document discovered while crawling the menu tree."""

    doc_id: str  # stable id: sha1 of source_url
    section_path: str  # breadcrumb, e.g. "Traffic Commercial Directorate > Commercial Circulars > 2019"
    title: str
    date_raw: Optional[str] = None
    date_parsed: Optional[str] = None  # ISO 8601 (YYYY-MM-DD) if parseable
    source_url: str = ""
    local_path: Optional[str] = None
    sha1: Optional[str] = None  # hash of the downloaded file bytes
    doc_type: str = ""  # pdf, doc, docx, xls, xlsx, unknown
    menu_id: str = ""  # the view_section.jsp id= this doc was listed under
    discovered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    download_ok: bool = False
    http_status: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


CSV_FIELDS = [
    "doc_id",
    "section_path",
    "title",
    "date_raw",
    "date_parsed",
    "source_url",
    "local_path",
    "sha1",
    "doc_type",
    "menu_id",
    "discovered_at",
    "download_ok",
    "http_status",
    "error",
]


def write_index(records: list[DocumentRecord], out_dir: Path) -> None:
    """Write _index.csv and _index.json into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "_index.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())


def load_index(out_dir: Path) -> list[DocumentRecord]:
    """Load existing _index.json (if present) as DocumentRecord list."""
    json_path = out_dir / "_index.json"
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [DocumentRecord(**item) for item in raw]
