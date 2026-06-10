"""Tests for image utility helpers."""

from pathlib import Path

import numpy as np
from PIL import Image

from acmp.utils.image import (
    cv2_to_pil,
    crop_panel,
    is_color_image,
    is_image_file,
    load_image,
    pil_to_cv2,
    resize_to_fit,
)


def test_pil_cv2_roundtrip():
    arr = np.zeros((30, 40, 3), dtype=np.uint8)
    arr[..., 0] = 200  # red in RGB
    img = Image.fromarray(arr, "RGB")
    cv = pil_to_cv2(img)
    assert cv.shape == (30, 40, 3)
    assert cv[0, 0, 2] == 200  # BGR -> red lands in channel 2
    back = cv2_to_pil(cv)
    assert np.array_equal(np.array(back), arr)


def test_resize_to_fit_exact_size_and_letterbox():
    img = Image.new("RGB", (100, 50), (255, 255, 255))
    out = resize_to_fit(img, 200, 200)
    assert out.size == (200, 200)
    assert out.getpixel((0, 0)) == (0, 0, 0)  # letterbox corner is black


def test_crop_panel():
    img = Image.new("RGB", (100, 100))
    crop = crop_panel(img, (10, 10, 50, 40))
    assert crop.size == (50, 40)


def test_is_image_file():
    assert is_image_file(Path("a.PNG"))
    assert is_image_file(Path("b.jpeg"))
    assert not is_image_file(Path("c.txt"))


def test_is_color_image(color_image, gray_image):
    assert is_color_image(color_image) is True
    assert is_color_image(gray_image) is False


def test_load_image_converts_to_rgb(tmp_path):
    p = tmp_path / "x.png"
    Image.new("L", (20, 20), 128).save(p)  # grayscale on disk
    img = load_image(p)
    assert img.mode == "RGB"
