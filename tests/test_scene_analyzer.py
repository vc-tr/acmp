"""Tests for LLM scene-analysis parsing and fallback (no network)."""

import base64
import io
import json

import pytest
from PIL import Image

from acmp.scene.analyzer import (
    PanelAnalysis,
    _image_to_base64,
    analyze_chapter,
    analyze_panel,
    parse_analysis_json,
)


def test_parse_plain_json():
    assert parse_analysis_json('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    assert parse_analysis_json('```json\n{"a": 1, "b": "x"}\n```') == {"a": 1, "b": "x"}


def test_parse_fenced_plain():
    assert parse_analysis_json("```\n{\"a\": 2}\n```") == {"a": 2}


def test_parse_prose_wrapped():
    txt = 'Sure! Here is the analysis:\n{"a": 3}\nHope that helps.'
    assert parse_analysis_json(txt) == {"a": 3}


def test_parse_garbage_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_analysis_json("not json at all")


def test_panel_analysis_from_dict():
    a = PanelAnalysis.from_dict(
        {
            "description": "d",
            "motion_intensity": "high",
            "characters": ["x"],
            "transition_to_next": "cut",
        }
    )
    assert a.description == "d"
    assert a.motion_intensity == "high"
    assert a.characters == ["x"]
    assert a.transition_to_next == "cut"


def test_panel_analysis_fallback():
    a = PanelAnalysis.fallback()
    assert a.motion_intensity == "low"
    assert a.transition_to_next == "crossfade"
    assert a.motion_prompt  # non-empty


def test_analyze_panel_fallback_no_network(color_image):
    a = analyze_panel(color_image, prefer="fallback")
    assert isinstance(a, PanelAnalysis)
    assert a.description


def test_analyze_chapter_fallback(color_image):
    res = analyze_chapter([color_image, color_image], prefer="fallback")
    assert len(res) == 2
    assert all(isinstance(r, PanelAnalysis) for r in res)


def test_image_to_base64_decodes_to_png(color_image):
    raw = base64.standard_b64decode(_image_to_base64(color_image))
    assert Image.open(io.BytesIO(raw)).format == "PNG"


def test_image_to_base64_resizes_large():
    big = Image.new("RGB", (2000, 1000), (10, 20, 30))
    raw = base64.standard_b64decode(_image_to_base64(big, max_size=512))
    assert max(Image.open(io.BytesIO(raw)).size) <= 512
