"""Tests for the Ken Burns animation renderer."""

from PIL import Image

from acmp.animation.ken_burns import render_ken_burns_frames


def _panel():
    return Image.new("RGB", (200, 300), (120, 130, 140))


def test_frame_count_and_size():
    frames = render_ken_burns_frames(_panel(), num_frames=6, output_size=(180, 320))
    assert len(frames) == 6
    assert all(f.size == (180, 320) for f in frames)
    assert all(f.mode == "RGB" for f in frames)


def test_all_effects_render_correct_size():
    effects = ("zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "unknown")
    for eff in effects:
        frames = render_ken_burns_frames(_panel(), num_frames=4, output_size=(100, 150), effect=eff)
        assert len(frames) == 4
        assert all(f.size == (100, 150) for f in frames)


def test_single_frame_does_not_crash():
    frames = render_ken_burns_frames(_panel(), num_frames=1, output_size=(100, 100))
    assert len(frames) == 1
