"""Pipeline configuration management."""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


@dataclass
class InputConfig:
    type: str = "auto"
    reading_order: str = "auto"
    dpi: int = 200


@dataclass
class PanelConfig:
    method: str = "contour"
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.95
    padding: int = 5


@dataclass
class DepthConfig:
    model: str = "midas_small"
    device: str = "auto"


@dataclass
class LayerConfig:
    method: str = "sam2"
    num_layers: int = 3
    inpaint_method: str = "opencv"


@dataclass
class ParallaxConfig:
    amplitude: float = 20.0
    frequency: float = 0.3
    direction: str = "horizontal"


@dataclass
class KenBurnsConfig:
    zoom_range: list[float] = field(default_factory=lambda: [1.0, 1.15])
    pan_speed: float = 30.0


@dataclass
class AnimationConfig:
    seconds_per_panel: float = 4.0
    transition_duration: float = 0.8
    parallax: ParallaxConfig = field(default_factory=ParallaxConfig)
    ken_burns: KenBurnsConfig = field(default_factory=KenBurnsConfig)


@dataclass
class OutputConfig:
    resolution: list[int] = field(default_factory=lambda: [1080, 1920])
    fps: int = 24
    codec: str = "libx264"
    bitrate: str = "5M"
    aspect_ratio: str = "9:16"


@dataclass
class AIMotionConfig:
    enabled: bool = False
    model: str = "wan2.1"


@dataclass
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    panels: PanelConfig = field(default_factory=PanelConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    layers: LayerConfig = field(default_factory=LayerConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ai_motion: AIMotionConfig = field(default_factory=AIMotionConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> PipelineConfig:
        """Load config, falling back to defaults."""
        if config_path and config_path.exists():
            return cls.from_yaml(config_path)
        if DEFAULT_CONFIG_PATH.exists():
            return cls.from_yaml(DEFAULT_CONFIG_PATH)
        return cls()

    @classmethod
    def _from_dict(cls, data: dict) -> PipelineConfig:
        """Build config from a dictionary."""
        cfg = cls()
        if "input" in data:
            cfg.input = InputConfig(**data["input"])
        if "panels" in data:
            cfg.panels = PanelConfig(**data["panels"])
        if "depth" in data:
            cfg.depth = DepthConfig(**data["depth"])
        if "layers" in data:
            d = data["layers"]
            cfg.layers = LayerConfig(**d)
        if "animation" in data:
            anim = data["animation"]
            parallax = ParallaxConfig(**anim.pop("parallax", {}))
            ken_burns = KenBurnsConfig(**anim.pop("ken_burns", {}))
            cfg.animation = AnimationConfig(**anim, parallax=parallax, ken_burns=ken_burns)
        if "output" in data:
            cfg.output = OutputConfig(**data["output"])
        if "ai_motion" in data:
            cfg.ai_motion = AIMotionConfig(**data["ai_motion"])
        return cfg
