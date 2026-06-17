"""Tests for FFmpeg video assembly."""

from pathlib import Path

import pytest
from PIL import Image

from acmp.video.assembler import _find_ffmpeg, frames_to_video


def test_empty_frames_raises(tmp_path):
    with pytest.raises(ValueError):
        frames_to_video([], tmp_path / "x.mp4")


def test_find_ffmpeg_returns_path():
    exe = _find_ffmpeg()
    assert Path(exe).exists() or exe == "ffmpeg"


@pytest.mark.slow
def test_encode_small_video(tmp_path):
    frames = [Image.new("RGB", (64, 64), (i * 20 % 255, 0, 0)) for i in range(6)]
    out = frames_to_video(frames, tmp_path / "v.mp4", fps=6)
    assert out.exists() and out.stat().st_size > 0
