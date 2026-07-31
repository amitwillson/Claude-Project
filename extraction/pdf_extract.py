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
        ocr_text, ocr_confidence = _extract_with_ocr(path, page_count)
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


# Poppler/Tesseract can occasionally wedge indefinitely on a malformed page,
# with no default timeout -- one bad page would otherwise hang an entire
# multi-thousand-document batch run forever. Rendering and OCR-ing one page
# at a time (rather than the whole document in one shot) means a genuinely
# large document (100+ pages) just takes proportionally longer instead of
# hitting one fixed cliff, while a single stuck/malformed page only costs
# that page -- extraction continues with the rest of the document instead
# of failing it entirely.
_PAGE_RENDER_TIMEOUT_SECONDS = 60  # rendering a single page to an image
_OCR_TIMEOUT_SECONDS = 60  # OCR-ing a single rendered page

# Some circulars are issued in Hindi (Devanagari script) or bilingually.
# Tesseract only recognizes English by default; without the "hin" language
# pack, Hindi text OCRs to near-nothing regardless of image quality. Detected
# once per process and cached, since the Hindi traineddata may not be
# installed on every machine -- fall back to English-only rather than
# failing every OCR call.
_ocr_lang_cache: Optional[str] = None


def _ocr_lang(pytesseract_module) -> str:
    global _ocr_lang_cache
    if _ocr_lang_cache is None:
        try:
            available = pytesseract_module.get_languages()
            _ocr_lang_cache = "eng+hin" if "hin" in available else "eng"
        except Exception:
            _ocr_lang_cache = "eng"
    return _ocr_lang_cache


# Poppler's default render DPI (~200) is often too low to recognize small
# print or faded old scans; rendering sharper images gives Tesseract more
# pixels per character to work with.
_OCR_DPI = 300


def _preprocess_for_ocr(image):
    """Grayscale + autocontrast before OCR -- cheap, no extra dependencies
    (PIL ships with pdf2image), and frequently recovers faded/low-contrast
    scans that Tesseract otherwise reads as blank or garbled."""
    from PIL import Image, ImageOps

    # PIL warns about "decompression bomb" risk on very large images (some
    # oversized tariff-table pages render to 100M+ pixels at 300 DPI). That
    # guard exists for untrusted/adversarial image sources; every image here
    # comes from a PDF we ourselves downloaded from the Railway Board site,
    # so it's a false positive worth silencing rather than spamming the log
    # on every large page.
    Image.MAX_IMAGE_PIXELS = None

    gray = image.convert("L")
    return ImageOps.autocontrast(gray)


def _extract_with_ocr(path: Path, page_count: int) -> tuple[str, float]:
    """Render each PDF page to an image and OCR it with pytesseract, one page
    at a time. A page that fails to render or OCR in time is skipped (its
    text stays empty) rather than aborting the whole document."""
    import pytesseract
    from pdf2image import convert_from_bytes

    lang = _ocr_lang(pytesseract)
    # Read the bytes ourselves and pipe them to Poppler rather than passing a
    # file path: some circular filenames + folder nesting exceed Windows'
    # classic 260-character MAX_PATH limit, which Python itself handles fine
    # but pdftoppm.exe (a native subprocess) cannot open.
    pdf_bytes = path.read_bytes()
    all_text = []
    confidences: list[float] = []
    for page_num in range(1, max(page_count, 1) + 1):
        try:
            images = convert_from_bytes(
                pdf_bytes,
                first_page=page_num,
                last_page=page_num,
                dpi=_OCR_DPI,
                timeout=_PAGE_RENDER_TIMEOUT_SECONDS,
            )
            image = _preprocess_for_ocr(images[0])
            data = pytesseract.image_to_data(
                image, lang=lang, output_type=pytesseract.Output.DICT, timeout=_OCR_TIMEOUT_SECONDS
            )
        except Exception:
            all_text.append("")  # this page timed out / failed to render -- skip it, keep going
            continue

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
