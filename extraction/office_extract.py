# DOC/DOCX/XLS/XLSX text extraction. DOCX/XLSX use well-supported libraries
# (python-docx, openpyxl); old binary .doc/.xls formats have no good pure-
# Python reader, so extraction there is best-effort (empty text + a flag).
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OfficeExtractionResult:
    text: str
    word_count: int
    error: Optional[str] = None
    best_effort: bool = False  # True for legacy binary .doc/.xls we can't parse well


def extract_office_text(path: str | Path) -> OfficeExtractionResult:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            return _extract_docx(path)
        if suffix in (".xlsx", ".xls"):
            if suffix == ".xls":
                return _legacy_unsupported(path, kind="xls")
            return _extract_xlsx(path)
        if suffix == ".doc":
            return _legacy_unsupported(path, kind="doc")
    except Exception as exc:
        return OfficeExtractionResult(text="", word_count=0, error=str(exc))

    return OfficeExtractionResult(text="", word_count=0, error=f"unsupported extension {suffix}")


def _extract_docx(path: Path) -> OfficeExtractionResult:
    import docx  # python-docx

    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text)
    text = "\n\n".join(paragraphs)
    return OfficeExtractionResult(text=text, word_count=len(text.split()))


def _extract_xlsx(path: Path) -> OfficeExtractionResult:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    lines = []
    for sheet in wb.worksheets:
        lines.append(f"--- Sheet: {sheet.title} ---")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                lines.append(" | ".join(cells))
    text = "\n".join(lines)
    return OfficeExtractionResult(text=text, word_count=len(text.split()))


def _legacy_unsupported(path: Path, kind: str) -> OfficeExtractionResult:
    # Old binary Office formats (pre-2007) have no reliable pure-Python
    # extractor available in this stack. Flag as best-effort/needs-review
    # rather than silently returning nothing with no signal.
    return OfficeExtractionResult(
        text="",
        word_count=0,
        error=f"legacy binary .{kind} format not supported by extractor — needs manual review",
        best_effort=True,
    )
