"""Tests for reading-order detection and panel sorting."""

import numpy as np
from PIL import Image

from acmp.eval.synthetic import generate_comic_page
from acmp.utils.reading_order import (
    detect_reading_order,
    sort_panels_by_reading_order,
)


def _to_grayscale_rgb(img):
    arr = np.array(img.convert("L"))
    return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")


def test_detect_vertical_by_aspect():
    page = generate_comic_page(4, 1, width=400, height=1400, seed=0)
    assert detect_reading_order([page.image]) == "vertical"


def test_detect_rtl_for_bw():
    page = generate_comic_page(2, 2, width=800, height=1000, seed=0)
    assert detect_reading_order([_to_grayscale_rgb(page.image)]) == "rtl"


def test_detect_ltr_for_color():
    page = generate_comic_page(2, 2, width=800, height=1000, seed=0)
    assert detect_reading_order([page.image]) == "ltr"


def test_detect_empty_defaults_ltr():
    assert detect_reading_order([]) == "ltr"


def test_sort_orders():
    boxes = [
        (0, 0, 100, 100),
        (200, 0, 100, 100),
        (0, 200, 100, 100),
        (200, 200, 100, 100),
    ]
    ltr = sort_panels_by_reading_order(boxes, "ltr", page_height=300)
    assert ltr == [
        (0, 0, 100, 100),
        (200, 0, 100, 100),
        (0, 200, 100, 100),
        (200, 200, 100, 100),
    ]
    rtl = sort_panels_by_reading_order(boxes, "rtl", page_height=300)
    assert rtl == [
        (200, 0, 100, 100),
        (0, 0, 100, 100),
        (200, 200, 100, 100),
        (0, 200, 100, 100),
    ]
    vert = sort_panels_by_reading_order(boxes, "vertical", page_height=300)
    assert [b[1] for b in vert] == [0, 0, 200, 200]


def test_sort_empty():
    assert sort_panels_by_reading_order([], "ltr", 100) == []
