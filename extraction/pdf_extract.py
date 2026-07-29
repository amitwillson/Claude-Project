# PDF text extraction with OCR fallback for scanned/image-only pages.
#
# Heavy dependencies (pypdf, pdfplumber, pytesseract, pdf2image) are imported
# lazily inside functions so this module can be imported (and unit-tested for
# the pure-Python bits) even in environments where those packages aren't
# installed yet.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Below this many words per page, a PDF is treated as having no usable text
# layer and OCR is attempted instead.
MIN_WORDS_PER_PAGE_FOR_TEXT_LAYER = 20


@dataclass
class ExtractionResult:
    text: str
    page_count: int
    word_count: int
    has_text_layer: bool
    ocr_used: bool
    ocr_confidence: Optional[float] = None  # average OCR confidence 0-100, if OCR used
    error: Optional[str] = None


def extract_pdf_text(path: str | Path) -> ExtractionResult:
    """Extract text from a PDF, falling back to OCR per-page if the native
    text layer looks empty/sparse (common in scanned older circulars)."""
    path = Path(path)
    try:
        text, page_count = _extract_with_pdfplumber(path)
    except Exception as exc:  # pdfplumber/pypdf missing or file unreadable
        return ExtractionResult(
            text="", page_count=0, word_count=0, has_text_layer=False,
            ocr_used=False, error=f"native extraction failed: {exc}",
        )

    word_count = len(text.split())
    words_per_page = word_count / page_count if page_count else 0
    has_text_layer = words_per_page >= MIN_WORDS_PER_PAGE_FOR_TEXT_LAYER

    if has_text_layer:
        return ExtractionResult(
            text=text, page_count=page_count, word_count=word_count,
            has_text_layer=True, ocr_used=False,
        )

    # Sparse/empty text layer -> OCR fallback.
    try:
        ocr_text, ocr_confidence = _extract_with_ocr(path)
    except Exception as exc:
        # OCR unavailable/failed: return whatever native text we found, flagged.
        return ExtractionResult(
            text=text, page_count=page_count, word_count=word_count,
            has_text_layer=False, ocr_used=False,
            error=f"OCR fallback failed: {exc}",
        )

    ocr_word_count = len(ocr_text.split())
    if ocr_word_count > word_count:
        return ExtractionResult(
            text=ocr_text, page_count=page_count, word_count=ocr_word_count,
            has_text_layer=False, ocr_used=True, ocr_confidence=ocr_confidence,
        )
    # OCR didn't help; keep the native (possibly empty) text but note OCR ran.
    return ExtractionResult(
        text=text, page_count=page_count, word_count=word_count,
        has_text_layer=False, ocr_used=True, ocr_confidence=ocr_confidence,
    )


def _extract_with_pdfplumber(path: Path) -> tuple[str, int]:
    import pdfplumber

    pages_text = []
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return "\n\n".join(pages_text), page_count


def _extract_with_ocr(path: Path) -> tuple[str, float]:
    """Render each PDF page to an image and OCR it with pytesseract."""
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(str(path))
    all_text = []
    confidences: list[float] = []
    for image in images:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        page_words = []
        for i, word in enumerate(data.get("text", [])):
            word = word.strip()
            if not word:
                continue
            page_words.append(word)
            conf = data.get("conf", [])[i] if i < len(data.get("conf", [])) else -1
            try:
                conf_f = float(conf)
                if conf_f >= 0:
                    confidences.append(conf_f)
            except (TypeError, ValueError):
                pass
        all_text.append(" ".join(page_words))
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return "\n\n".join(all_text), avg_conf


def flag_extraction_outliers(result: ExtractionResult, words_per_page_threshold: int = 30) -> bool:
    """Return True if the extraction looks suspicious and needs manual review:
    a document with real pages but a suspiciously low word count (e.g. a
    10-page circular that yielded 40 words almost certainly lost content)."""
    if result.error:
        return True
    if result.page_count <= 0:
        return True
    words_per_page = result.word_count / result.page_count
    if words_per_page < words_per_page_threshold:
        return True
    if result.ocr_used and result.ocr_confidence is not None and result.ocr_confidence < 50:
        return True
    return False
