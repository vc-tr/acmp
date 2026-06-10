"""Tests for the config-driven panel-detector factory."""

from acmp.config import PanelConfig
from acmp.panels.detector import make_panel_detector


def test_default_uses_contour(grid_page):
    det = make_panel_detector(PanelConfig())  # method="contour"
    assert len(det(grid_page.image)) == 4


def test_yolo_without_weights_falls_back(grid_page):
    det = make_panel_detector(PanelConfig(method="yolo", weights=None))
    assert len(det(grid_page.image)) == 4


def test_yolo_bad_weights_falls_back(grid_page, tmp_path):
    det = make_panel_detector(
        PanelConfig(method="yolo", weights=str(tmp_path / "missing.pt"))
    )
    assert len(det(grid_page.image)) == 4  # load fails -> heuristic fallback
