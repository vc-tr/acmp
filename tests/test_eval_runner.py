"""Tests for the detector benchmark runner."""

from acmp.eval.metrics import DetectionMetrics
from acmp.eval.runner import (
    benchmark,
    benchmark_suite,
    format_results,
    heuristic_detector,
)
from acmp.eval.synthetic import generate_dataset


def test_heuristic_detector_returns_boxes_and_scores(grid_page):
    boxes, scores = heuristic_detector(grid_page.image)
    assert len(boxes) == len(scores) == 4
    assert all(s > 0 for s in scores)


def test_benchmark_clean_is_strong():
    pages = generate_dataset(n=6, seed=0)
    m = benchmark(heuristic_detector, pages, iou_threshold=0.5)
    assert isinstance(m, DetectionMetrics)
    assert m.recall == 1.0  # heuristic finds every panel on clean pages


def test_benchmark_suite_structure():
    pages = generate_dataset(n=4, seed=1)
    results = benchmark_suite({"h": heuristic_detector}, {"clean": pages})
    assert isinstance(results["clean"]["h"], DetectionMetrics)


def test_format_results_markdown():
    pages = generate_dataset(n=3, seed=2)
    results = benchmark_suite({"h": heuristic_detector}, {"clean": pages})
    table = format_results(results)
    assert "| Condition | Detector |" in table
    assert "clean" in table and "| h |" in table
