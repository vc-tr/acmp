"""Extract page images from PDF files."""

from __future__ import annotations

import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def extract_pages_from_pdf(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Extract each page of a PDF as a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering pages (higher = better quality, slower).

    Returns:
        List of PIL Images, one per page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. Install with: pip install PyMuPDF"
        )

    doc = fitz.open(str(pdf_path))
    pages = []

    zoom = dpi / 72.0  # PDF standard is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(img)
        logger.debug(f"Extracted PDF page {page_num + 1}/{len(doc)}: {pix.width}x{pix.height}")

    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
    return pages
