import os
from pathlib import Path

import pdfplumber

from matcher.models import SourceType


def _configure_tesseract() -> None:
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".tif"}


def extract_native_text(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    return "\n".join(pages)


def extract_ocr_text(path: Path, dpi: int = 300) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("pytesseract or Pillow not installed.")

    _configure_tesseract()

    path = Path(path)

    if path.suffix.lower() in _IMAGE_SUFFIXES:
        img = Image.open(path)
        return pytesseract.image_to_string(img, config="--psm 6")

    doc = fitz.open(str(path))
    pages_text = []
    try:
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, config="--psm 6")
            if text.strip():
                pages_text.append(text)
    finally:
        doc.close()
    return "\n".join(pages_text)


def extract_text(path: Path, source_type: SourceType) -> tuple[str, str]:
    """Return (text, method_used)."""
    if source_type == SourceType.NATIVE_PDF:
        return extract_native_text(path), "native"
    return extract_ocr_text(path), "tesseract"
