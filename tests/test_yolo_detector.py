"""Tests for the YOLO detector wrapper (no trained weights needed)."""

import pytest

from acmp.panels.yolo_detector import _resolve_device


def test_resolve_device_explicit():
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_auto():
    # Returns whatever backend is available; never raises.
    assert _resolve_device("auto") in ("mps", "cpu", "0")


def test_missing_weights_raises(tmp_path):
    pytest.importorskip("ultralytics")
    from acmp.panels.yolo_detector import YoloPanelDetector

    with pytest.raises(FileNotFoundError):
        YoloPanelDetector(tmp_path / "does_not_exist.pt")
