"""Tests for AI-frame post-processing (color match, sharpen, upscale)."""

from PIL import Image

from acmp.animation.postprocess import (
    match_color_histogram,
    sharpen_frame,
    upscale_two_stage,
)


def _img(w, h, color=(100, 120, 140)):
    return Image.new("RGB", (w, h), color)


def test_match_color_histogram_size_and_mode():
    out = match_color_histogram(_img(64, 64, (200, 50, 50)), _img(64, 64, (50, 50, 200)))
    assert out.size == (64, 64) and out.mode == "RGB"


def test_sharpen_frame_preserves_size():
    assert sharpen_frame(_img(48, 64)).size == (48, 64)


def test_upscale_two_stage_reaches_target():
    assert upscale_two_stage(_img(100, 150), 400, 600).size == (400, 600)


def test_upscale_two_stage_when_already_large():
    assert upscale_two_stage(_img(800, 1200), 400, 600).size == (400, 600)
