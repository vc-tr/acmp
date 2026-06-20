"""Tests for background inpainting (acmp/layers/inpainter.py)."""

import numpy as np
import pytest
from PIL import Image

from acmp.layers.inpainter import inpaint_background


def _synthetic_image_and_mask(size: int = 32):
    img = Image.new("RGB", (size, size), (120, 90, 60))
    mask = np.zeros((size, size), dtype=bool)
    mask[8:16, 8:16] = True  # foreground block to fill
    return img, mask


def test_opencv_inpaint_returns_same_size_rgb():
    img, mask = _synthetic_image_and_mask()
    out = inpaint_background(img, mask, method="opencv")
    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert out.mode == "RGB"


def test_lama_falls_back_to_opencv_when_unavailable():
    # simple-lama-inpainting is not a CI dependency, so the 'lama' path must
    # degrade gracefully (return a valid image) rather than raise.
    img, mask = _synthetic_image_and_mask()
    out = inpaint_background(img, mask, method="lama")
    assert isinstance(out, Image.Image)
    assert out.size == img.size


def test_unknown_method_raises():
    img, mask = _synthetic_image_and_mask()
    with pytest.raises(ValueError):
        inpaint_background(img, mask, method="bogus")
