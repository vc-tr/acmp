"""Benchmark panel detectors against ground truth on labeled page sets.

A *detector* here is any callable ``image -> (boxes, scores)`` where boxes are
``(x, y, w, h)`` pixel tuples and scores are per-box confidences (used for AP
ranking). This uniform interface lets us compare the OpenCV heuristic and a
learned YOLO model with the exact same evaluation code.
"""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from acmp.eval.metrics import Box, DetectionMetrics, evaluate_detections
from acmp.eval.synthetic import SyntheticPage
from acmp.panels.detector import detect_panels

Detector = Callable[[Image.Image], "tuple[list[Box], list[float]]"]


def heuristic_detector(image: Image.Image) -> tuple[list[Box], list[float]]:
    """Adapter for the OpenCV contour detector.

    It has no notion of confidence, so we use box area as a proxy score for AP
    ranking (larger panels first). Treat its AP as indicative.
    """
    boxes = list(detect_panels(image))
    scores = [float(w * h) for (_, _, w, h) in boxes]
    return boxes, scores


def benchmark(detector: Detector, pages: list[SyntheticPage], iou_threshold: float = 0.5) -> DetectionMetrics:
    """Run a detector over labeled pages and return aggregate metrics."""
    all_preds, all_scores, all_gts = [], [], []
    for page in pages:
        boxes, scores = detector(page.image)
        all_preds.append(boxes)
        all_scores.append(scores)
        all_gts.append(page.boxes)
    return evaluate_detections(all_preds, all_gts, all_scores, iou_threshold)


def benchmark_suite(
    detectors: dict[str, Detector],
    conditions: dict[str, list[SyntheticPage]],
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, DetectionMetrics]]:
    """Benchmark every detector under every condition.

    Returns ``results[condition][detector_name] -> DetectionMetrics``.
    """
    results: dict[str, dict[str, DetectionMetrics]] = {}
    for cond, pages in conditions.items():
        results[cond] = {
            name: benchmark(fn, pages, iou_threshold) for name, fn in detectors.items()
        }
    return results


def format_results(results: dict[str, dict[str, DetectionMetrics]]) -> str:
    """Render benchmark results as a Markdown table."""
    lines = [
        "| Condition | Detector | Precision | Recall | F1 | AP@0.5 | mIoU |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond, by_det in results.items():
        for name, m in by_det.items():
            lines.append(
                f"| {cond} | {name} | {m.precision:.3f} | {m.recall:.3f} | "
                f"{m.f1:.3f} | {m.ap50:.3f} | {m.mean_iou:.3f} |"
            )
    return "\n".join(lines)
