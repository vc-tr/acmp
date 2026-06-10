"""Tests for the synthetic comic-page generator."""

import numpy as np
import pytest

from acmp.eval.metrics import box_iou
from acmp.eval.synthetic import (
    boxes_to_yolo,
    degrade_image,
    generate_comic_page,
    generate_dataset,
    generate_split,
    yolo_to_boxes,
)


def test_generate_grid_count_and_size():
    page = generate_comic_page(rows=2, cols=3, width=900, height=1200, seed=0)
    assert len(page.boxes) == 6
    assert page.image.size == (900, 1200)


def test_boxes_within_page_and_disjoint():
    page = generate_comic_page(rows=2, cols=2, width=800, height=1000)
    w, h = page.image.size
    for (x, y, bw, bh) in page.boxes:
        assert 0 <= x and 0 <= y and x + bw <= w and y + bh <= h
    for i in range(len(page.boxes)):
        for j in range(i + 1, len(page.boxes)):
            assert box_iou(page.boxes[i], page.boxes[j]) == 0.0


def test_unpacking():
    img, boxes = generate_comic_page(1, 1)
    assert img.size[0] > 0 and len(boxes) == 1


def test_invalid_dims_raise():
    with pytest.raises(ValueError):
        generate_comic_page(rows=0, cols=2)


def test_too_large_margin_raises():
    with pytest.raises(ValueError):
        generate_comic_page(rows=2, cols=2, width=100, height=100, margin=60)


def test_dataset_size_and_labels():
    ds = generate_dataset(n=8, seed=1)
    assert len(ds) == 8
    assert all(len(p.boxes) >= 1 for p in ds)


def test_yolo_roundtrip():
    page = generate_comic_page(2, 2, 800, 1000, seed=5)
    lines = boxes_to_yolo(page.boxes, 800, 1000)
    assert len(lines) == 4
    assert all(line.startswith("0 ") for line in lines)
    back = yolo_to_boxes(lines, 800, 1000)
    for orig, rt in zip(page.boxes, back):
        assert all(abs(a - b) <= 1 for a, b in zip(orig, rt))


def test_degrade_preserves_geometry():
    page = generate_comic_page(2, 2, 600, 800, seed=0)
    deg = degrade_image(page.image, seed=1)
    assert deg.size == page.image.size
    assert deg.mode == "RGB"


def test_generate_split_keys_and_counts():
    split = generate_split(n_train=5, n_val=3, n_test=4, seed=0)
    assert set(split) == {"train", "val", "test_clean", "test_degraded"}
    assert len(split["train"]) == 5
    assert len(split["val"]) == 3
    assert len(split["test_clean"]) == 4
    assert len(split["test_degraded"]) == 4


def test_degraded_differs_from_clean_but_keeps_boxes():
    split = generate_split(n_train=0, n_val=0, n_test=3, seed=0)
    clean, degraded = split["test_clean"][0], split["test_degraded"][0]
    a, b = np.asarray(clean.image), np.asarray(degraded.image)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)        # pixels were degraded
    assert clean.boxes == degraded.boxes   # geometry (labels) unchanged
