"""Tests for OpenCV panel detection."""

import math

from acmp.eval.metrics import match_boxes
from acmp.panels.detector import (
    _compute_iou,
    _remove_overlapping,
    detect_panels,
    detect_panels_vertical_scroll,
)


def test_compute_iou():
    assert math.isclose(_compute_iou((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3, rel_tol=1e-6)
    assert _compute_iou((0, 0, 10, 10), (100, 100, 5, 5)) == 0.0


def test_remove_overlapping_keeps_larger():
    big = (0, 0, 100, 100)
    dup = (5, 5, 95, 95)  # heavy overlap with big
    far = (300, 300, 50, 50)
    kept = _remove_overlapping([dup, big, far])
    assert big in kept
    assert far in kept
    assert len(kept) == 2


def test_detect_panels_grid(grid_page):
    preds = list(detect_panels(grid_page.image))
    assert len(preds) == 4
    idx, _ = match_boxes(preds, grid_page.boxes, iou_threshold=0.7)
    assert sum(1 for i in idx if i >= 0) == 4  # all GT panels found at high IoU


def test_detect_vertical_scroll(tall_page):
    preds = detect_panels_vertical_scroll(tall_page.image)
    assert len(preds) == 5
