"""Tests for pipeline configuration loading and overrides."""

import yaml

from acmp.config import DEFAULT_CONFIG_PATH, PipelineConfig


def test_defaults():
    cfg = PipelineConfig()
    assert cfg.output.resolution == [1080, 1920]
    assert cfg.output.fps == 24
    assert cfg.animation.seconds_per_panel == 4.0
    assert cfg.input.reading_order == "auto"


def test_load_default_yaml_exists():
    assert DEFAULT_CONFIG_PATH.exists()
    cfg = PipelineConfig.load()
    assert cfg.output.resolution == [1080, 1920]


def test_from_dict_nested_animation():
    data = {
        "animation": {
            "seconds_per_panel": 2.5,
            "transition_duration": 0.5,
            "parallax": {"amplitude": 10.0},
            "ken_burns": {"pan_speed": 50.0},
        }
    }
    cfg = PipelineConfig._from_dict(data)
    assert cfg.animation.seconds_per_panel == 2.5
    assert cfg.animation.parallax.amplitude == 10.0
    assert cfg.animation.ken_burns.pan_speed == 50.0


def test_from_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"output": {"fps": 12, "resolution": [720, 1280]}}))
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.output.fps == 12
    assert cfg.output.resolution == [720, 1280]


def test_load_missing_falls_back_to_default(tmp_path):
    cfg = PipelineConfig.load(tmp_path / "does_not_exist.yaml")
    assert cfg.output.fps == 24
