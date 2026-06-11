"""Tests for Wan VACE support code that needs no GPU / model weights."""

import pytest
from PIL import Image

from acmp.animation import wan_animator as wa


def _img(w=200, h=300):
    return Image.new("RGB", (w, h), (120, 130, 140))


def test_prepare_image_rounds_to_multiple_of_16():
    out = wa._prepare_image(_img(200, 300), 321, 577)
    assert out.size == (320, 576)  # rounded down to multiples of 16
    assert out.mode == "RGB"


def test_create_vace_inputs_counts_and_modes():
    video, masks = wa._create_vace_inputs(_img(), num_frames=13, width=256, height=448)
    assert len(video) == len(masks) == 13
    assert all(f.size == (256, 448) for f in video)
    assert all(m.mode == "L" and m.size == (256, 448) for m in masks)
    assert masks[0].getpixel((0, 0)) == 0      # first frame kept
    assert masks[1].getpixel((0, 0)) == 255    # later frames generated


def test_quality_presets_ordered():
    assert wa.QUALITY_PRESETS["fast"] < wa.QUALITY_PRESETS["balanced"] < wa.QUALITY_PRESETS["quality"]


def test_safe_returns_frames_on_success(monkeypatch):
    pytest.importorskip("torch")
    frames = [_img(320, 576)]
    monkeypatch.setattr(wa, "animate_panel", lambda **kw: frames)
    assert wa.animate_panel_safe(_img()) is frames


def test_safe_returns_none_on_oom(monkeypatch):
    pytest.importorskip("torch")

    def boom(**kw):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(wa, "animate_panel", boom)
    assert wa.animate_panel_safe(_img()) is None


def test_safe_returns_none_on_other_error(monkeypatch):
    pytest.importorskip("torch")

    def boom(**kw):
        raise ValueError("model download failed")

    monkeypatch.setattr(wa, "animate_panel", boom)
    assert wa.animate_panel_safe(_img()) is None
