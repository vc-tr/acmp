"""Tests for pipeline helpers and an end-to-end (no-AI) integration run."""

from pathlib import Path

import pytest
from PIL import Image

from acmp.config import PipelineConfig
from acmp.pipeline import _apply_transitions, _extend_frames_pingpong
from acmp.scene.analyzer import PanelAnalysis


def _frames(n, color=(0, 0, 0)):
    return [Image.new("RGB", (16, 24), color) for _ in range(n)]


def test_pingpong_extends():
    assert len(_extend_frames_pingpong(_frames(3), target_count=7)) == 7


def test_pingpong_truncates():
    assert len(_extend_frames_pingpong(_frames(10), target_count=4)) == 4


def test_pingpong_single_frame():
    assert len(_extend_frames_pingpong(_frames(1), target_count=5)) == 5


def test_pingpong_two_frames():
    # reverse-middle is empty here; must still reach the target length
    assert len(_extend_frames_pingpong(_frames(2), target_count=5)) == 5


def test_apply_transitions_cut_preserves_count():
    cfg = PipelineConfig()
    cfg.animation.transition_duration = 0.5
    a = PanelAnalysis.fallback()
    a.transition_to_next = "cut"
    b = PanelAnalysis.fallback()
    out = _apply_transitions([_frames(5), _frames(5)], [a, b], cfg, fps=4)
    assert len(out) == 10


def test_apply_transitions_empty():
    assert _apply_transitions([], [], PipelineConfig(), fps=4) == []


@pytest.mark.slow
def test_end_to_end_no_ai(tmp_path):
    from acmp.eval.synthetic import generate_comic_page
    from acmp.pipeline import process_chapter

    page = generate_comic_page(2, 2, width=600, height=800, seed=0)
    in_dir = tmp_path / "pages"
    in_dir.mkdir()
    page.image.save(in_dir / "p001.png")

    cfg = PipelineConfig()
    cfg.output.resolution = [240, 320]
    cfg.output.fps = 4
    cfg.animation.seconds_per_panel = 0.5
    cfg.animation.transition_duration = 0.25

    out = tmp_path / "out.mp4"
    result = process_chapter(in_dir, out, config=cfg, use_ai=False, llm_prefer="fallback")
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0
