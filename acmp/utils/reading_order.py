"""Reading order detection for comics, manga, and manhwa."""

from __future__ import annotations

from PIL import Image
from acmp.utils.image import is_color_image


def detect_reading_order(pages: list[Image.Image]) -> str:
    """Auto-detect the reading order from a list of page images.

    Returns:
        'rtl' for manga (right-to-left, typically B&W)
        'ltr' for western comics (left-to-right, typically color)
        'vertical' for manhwa/webtoon (long vertical strips, color)
    """
    if not pages:
        return "ltr"

    # Sample a few pages for heuristics
    sample = pages[:min(3, len(pages))]

    # Check aspect ratios - manhwa pages are very tall (vertical scroll)
    avg_aspect = sum(p.width / max(p.height, 1) for p in sample) / len(sample)
    if avg_aspect < 0.4:
        return "vertical"

    # Check if pages are predominantly color or B&W
    color_count = sum(1 for p in sample if is_color_image(p))
    is_mostly_color = color_count > len(sample) / 2

    # Manga is typically B&W; western comics and manhwa are color
    # If page-based (not vertical) and B&W, assume manga (RTL)
    if not is_mostly_color:
        return "rtl"

    return "ltr"


def sort_panels_by_reading_order(
    panels: list[tuple[int, int, int, int]],
    reading_order: str,
    page_height: int,
) -> list[tuple[int, int, int, int]]:
    """Sort panel bounding boxes according to reading order.

    Args:
        panels: List of (x, y, w, h) bounding boxes.
        reading_order: 'rtl', 'ltr', or 'vertical'.
        page_height: Height of the page (for row grouping).

    Returns:
        Sorted list of panel bounding boxes.
    """
    if not panels:
        return panels

    if reading_order == "vertical":
        # Top to bottom
        return sorted(panels, key=lambda p: p[1])

    # Group panels into rows based on vertical overlap
    row_threshold = page_height * 0.05  # 5% of page height
    rows: list[list[tuple[int, int, int, int]]] = []
    sorted_by_y = sorted(panels, key=lambda p: p[1])

    current_row: list[tuple[int, int, int, int]] = [sorted_by_y[0]]
    current_row_y = sorted_by_y[0][1]

    for panel in sorted_by_y[1:]:
        if abs(panel[1] - current_row_y) < row_threshold:
            current_row.append(panel)
        else:
            rows.append(current_row)
            current_row = [panel]
            current_row_y = panel[1]
    rows.append(current_row)

    # Sort within each row
    result = []
    for row in rows:
        if reading_order == "rtl":
            row.sort(key=lambda p: p[0], reverse=True)
        else:
            row.sort(key=lambda p: p[0])
        result.extend(row)

    return result
