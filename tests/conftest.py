"""Shared pytest fixtures for the acmp test suite."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from acmp.eval.synthetic import generate_comic_page


@pytest.fixture
def grid_page():
    """A clean 2x2 synthetic comic page with 4 ground-truth panels."""
    return generate_comic_page(rows=2, cols=2, width=800, height=1000, seed=1)


@pytest.fixture
def tall_page():
    """A tall 5x1 page for vertical-scroll (manhwa) detection."""
    return generate_comic_page(rows=5, cols=1, width=600, height=2600, seed=3)


@pytest.fixture
def color_image():
    """A saturated colour image."""
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    arr[..., 0] = 220
    arr[..., 2] = 40
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def gray_image():
    """A flat gray (desaturated) image."""
    arr = np.full((120, 120, 3), 130, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def small_panel():
    """A small RGB panel for animation tests."""
    return generate_comic_page(rows=1, cols=1, width=300, height=420, seed=2).image
