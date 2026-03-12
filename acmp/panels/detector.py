"""Comic panel detection using OpenCV contour analysis."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import logging

from acmp.utils.image import pil_to_cv2
from acmp.config import PanelConfig

logger = logging.getLogger(__name__)


def detect_panels(
    page: Image.Image,
    config: PanelConfig | None = None,
) -> list[tuple[int, int, int, int]]:
    """Detect panel bounding boxes from a comic/manga page.

    Uses adaptive thresholding and contour detection to find rectangular
    panel regions. Works for both manga (B&W) and manhwa (color).

    Args:
        page: PIL Image of a full page.
        config: Panel detection configuration.

    Returns:
        List of (x, y, w, h) bounding boxes for each detected panel.
    """
    if config is None:
        config = PanelConfig()

    img = pil_to_cv2(page)
    h, w = img.shape[:2]
    page_area = h * w

    min_area = page_area * config.min_area_ratio
    max_area = page_area * config.max_area_ratio

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to handle varying backgrounds
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate to connect nearby edges (panel borders)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch

        # Filter by area
        if area < min_area or area > max_area:
            continue

        # Filter by aspect ratio (panels shouldn't be too thin)
        aspect = min(cw, ch) / max(cw, ch)
        if aspect < 0.1:
            continue

        # Apply padding
        pad = config.padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        cw = min(w - x, cw + 2 * pad)
        ch = min(h - y, ch + 2 * pad)

        panels.append((x, y, cw, ch))

    # Remove overlapping panels (keep larger ones)
    panels = _remove_overlapping(panels)

    logger.info(f"Detected {len(panels)} panels on page ({w}x{h})")
    return panels


def detect_panels_vertical_scroll(
    page: Image.Image,
    config: PanelConfig | None = None,
) -> list[tuple[int, int, int, int]]:
    """Detect panels in vertical scroll (manhwa/webtoon) format.

    For long vertical strips, this splits the image into logical sections
    based on horizontal whitespace gaps.

    Args:
        page: PIL Image of a vertical scroll page.
        config: Panel detection configuration.

    Returns:
        List of (x, y, w, h) bounding boxes, ordered top to bottom.
    """
    if config is None:
        config = PanelConfig()

    img = pil_to_cv2(page)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Project horizontally: find rows that are mostly white (gutters)
    row_mean = np.mean(gray, axis=1)
    white_threshold = 240

    is_gutter = row_mean > white_threshold
    panels = []
    panel_start = None

    for y_pos in range(h):
        if not is_gutter[y_pos]:
            if panel_start is None:
                panel_start = y_pos
        else:
            if panel_start is not None:
                panel_h = y_pos - panel_start
                min_h = h * config.min_area_ratio * 2  # Minimum panel height
                if panel_h > min_h:
                    panels.append((0, panel_start, w, panel_h))
                panel_start = None

    # Don't forget the last panel
    if panel_start is not None:
        panel_h = h - panel_start
        if panel_h > h * config.min_area_ratio * 2:
            panels.append((0, panel_start, w, panel_h))

    # If no gutters found, try contour-based detection as fallback
    if len(panels) <= 1:
        return detect_panels(page, config)

    logger.info(f"Detected {len(panels)} panels in vertical scroll ({w}x{h})")
    return panels


def _remove_overlapping(
    panels: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Remove overlapping panels using non-maximum suppression."""
    if not panels:
        return panels

    # Sort by area (largest first)
    panels = sorted(panels, key=lambda p: p[2] * p[3], reverse=True)
    keep = []

    for panel in panels:
        is_overlap = False
        for kept in keep:
            if _compute_iou(panel, kept) > iou_threshold:
                is_overlap = True
                break
        if not is_overlap:
            keep.append(panel)

    return keep


def _compute_iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """Compute Intersection over Union for two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
