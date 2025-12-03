from pathlib import Path
from unittest.mock import MagicMock, patch

from matcher.classify import classify_pdf
from matcher.models import SourceType


def test_native_pdf(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "TQL RATE CONFIRMATION FOR PO# 12345 " * 3
    with patch("matcher.classify.pdfplumber.open") as mo:
        mo.return_value.__enter__.return_value.pages = [mock_page]
        assert classify_pdf(pdf) == SourceType.NATIVE_PDF


def test_cid_garbage_is_clean_scan(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "(cid:2)(cid:3)(cid:0)(cid:28)" * 20
    with patch("matcher.classify.pdfplumber.open") as mo:
        mo.return_value.__enter__.return_value.pages = [mock_page]
        assert classify_pdf(pdf) == SourceType.CLEAN_SCAN


def test_empty_text_is_clean_scan(tmp_path):
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    with patch("matcher.classify.pdfplumber.open") as mo:
        mo.return_value.__enter__.return_value.pages = [mock_page]
        assert classify_pdf(pdf) == SourceType.CLEAN_SCAN


def test_phone_photo_jpg(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    mock_img = MagicMock()
    mock_img.size = (1080, 1920)  # 16:9 portrait — ratio 1.78, clearly not letter paper
    with patch("PIL.Image.open", return_value=mock_img):
        assert classify_pdf(img) == SourceType.PHONE_PHOTO


def test_standard_scan_ratio(tmp_path):
    img = tmp_path / "scan.png"
    img.write_bytes(b"\x89PNG")
    mock_img = MagicMock()
    mock_img.size = (2550, 3300)  # 8.5×11 at 300 DPI → ratio ~1.29
    with patch("PIL.Image.open", return_value=mock_img):
        assert classify_pdf(img) == SourceType.CLEAN_SCAN
