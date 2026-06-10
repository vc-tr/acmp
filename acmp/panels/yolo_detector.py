"""Learned panel detector (YOLOv8) — a drop-in alternative to the heuristic.

Wraps an ultralytics YOLO model behind the same ``(x, y, w, h)`` interface the
rest of acmp uses, so a trained detector can either be benchmarked against the
OpenCV heuristic or plugged into the pipeline via ``panels.method: yolo``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from acmp.eval.metrics import Box

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


class YoloPanelDetector:
    """A YOLOv8 panel detector exposing ``detect`` / ``detect_with_scores``."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for the YOLO detector. "
                'Install with: pip install -e ".[train]"'
            ) from e

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")

        self.device = _resolve_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = YOLO(str(weights))
        logger.info(f"Loaded YOLO panel detector from {weights} (device={self.device})")

    def detect_with_scores(self, image: Image.Image) -> tuple[list[Box], list[float]]:
        """Detect panels, returning ((x, y, w, h) boxes, confidence scores)."""
        result = self.model.predict(
            source=image.convert("RGB"),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        boxes: list[Box] = []
        scores: list[float] = []
        for xyxy, score in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = xyxy
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            scores.append(float(score))
        return boxes, scores

    def detect(self, image: Image.Image) -> list[Box]:
        """Detect panels, returning (x, y, w, h) boxes only."""
        return self.detect_with_scores(image)[0]

    def __call__(self, image: Image.Image) -> tuple[list[Box], list[float]]:
        return self.detect_with_scores(image)
