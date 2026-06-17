"""Evaluation utilities: synthetic data, detection metrics, benchmark runner.

This package lets us *quantitatively* evaluate panel detectors (heuristic vs.
learned) instead of eyeballing results — a core AI-engineering practice.
"""

from acmp.eval.metrics import (
    DetectionMetrics,
    average_precision,
    box_iou,
    evaluate_detections,
    match_boxes,
    precision_recall_f1,
)
from acmp.eval.synthetic import generate_comic_page, generate_dataset

__all__ = [
    "box_iou",
    "match_boxes",
    "precision_recall_f1",
    "average_precision",
    "evaluate_detections",
    "DetectionMetrics",
    "generate_comic_page",
    "generate_dataset",
]
