"""Tests for panel-to-panel transitions."""

import math

from PIL import Image

from acmp.animation.transitions import _ease_in_out, crossfade, slide_transition


def _frames(n, color):
    return [Image.new("RGB", (32, 48), color) for _ in range(n)]


def test_crossfade_length():
    out = crossfade(_frames(5, (0, 0, 0)), _frames(5, (255, 255, 255)), transition_frames=2)
    assert len(out) == 5 + 5 - 2


def test_crossfade_zero_is_concat():
    out = crossfade(_frames(3, (0, 0, 0)), _frames(3, (255, 255, 255)), transition_frames=0)
    assert len(out) == 6


def test_crossfade_clamps_to_available():
    out = crossfade(_frames(2, (0, 0, 0)), _frames(5, (255, 255, 255)), transition_frames=10)
    assert len(out) == 2 + 5 - 2  # clamped to min(10, 2, 5) = 2


def test_slide_transition_count_and_size():
    a = Image.new("RGB", (40, 60), (0, 0, 0))
    b = Image.new("RGB", (40, 60), (255, 255, 255))
    for direction in ("left", "right", "up", "down"):
        frames = slide_transition(a, b, num_frames=5, direction=direction)
        assert len(frames) == 5
        assert all(f.size == (40, 60) for f in frames)


def test_ease_in_out_bounds_and_monotonic():
    assert math.isclose(_ease_in_out(0.0), 0.0)
    assert math.isclose(_ease_in_out(1.0), 1.0)
    assert math.isclose(_ease_in_out(0.5), 0.5)
    vals = [_ease_in_out(t / 10) for t in range(11)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
