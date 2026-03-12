"""Unified input loader for images and PDFs."""

from __future__ import annotations

import logging
from pathlib import Path
from PIL import Image

from acmp.utils.image import is_image_file, load_image
from acmp.ingest.pdf_extractor import extract_pages_from_pdf

logger = logging.getLogger(__name__)


def load_chapter(input_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """Load chapter pages from a directory of images or a PDF file.

    Args:
        input_path: Path to a directory of images, a single PDF, or a single image.
        dpi: DPI for PDF rendering.

    Returns:
        Ordered list of page images (PIL RGB).
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    # Single PDF file
    if path.is_file() and path.suffix.lower() == ".pdf":
        logger.info(f"Loading PDF: {path}")
        return extract_pages_from_pdf(path, dpi=dpi)

    # Single image file
    if path.is_file() and is_image_file(path):
        logger.info(f"Loading single image: {path}")
        return [load_image(path)]

    # Directory of images
    if path.is_dir():
        image_files = sorted(
            [f for f in path.iterdir() if f.is_file() and is_image_file(f)],
            key=lambda f: f.name,
        )
        if not image_files:
            raise ValueError(f"No image files found in directory: {path}")

        logger.info(f"Loading {len(image_files)} images from {path}")
        pages = []
        for img_path in image_files:
            pages.append(load_image(img_path))
            logger.debug(f"  Loaded: {img_path.name}")
        return pages

    raise ValueError(
        f"Unsupported input: {path}. Provide a directory of images, a PDF file, or an image file."
    )
