"""Tests for detection metrics (IoU, matching, P/R/F1, AP)."""

import math

import pytest

from acmp.eval.metrics import (
    average_precision,
    box_iou,
    evaluate_detections,
    match_boxes,
    precision_recall_f1,
)


def test_box_iou_identical():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_box_iou_disjoint():
    assert box_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_box_iou_half_overlap():
    assert math.isclose(box_iou((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3, rel_tol=1e-6)


def test_match_boxes_perfect():
    gts = [(0, 0, 10, 10), (50, 50, 10, 10)]
    preds = [(0, 0, 10, 10), (50, 50, 10, 10)]
    idx, ious = match_boxes(preds, gts, 0.5)
    assert sorted(idx) == [0, 1]
    assert all(iou == 1.0 for iou in ious)


def test_match_boxes_each_gt_matched_once():
    gts = [(0, 0, 10, 10)]
    preds = [(0, 0, 10, 10), (0, 0, 10, 10)]  # two preds, one gt
    idx, _ = match_boxes(preds, gts, 0.5)
    assert sum(1 for i in idx if i >= 0) == 1  # only one TP


def test_precision_recall_f1():
    p, r, f = precision_recall_f1(tp=8, fp=2, fn=2)
    assert math.isclose(p, 0.8) and math.isclose(r, 0.8) and math.isclose(f, 0.8)


def test_precision_recall_f1_zero():
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_average_precision_perfect():
    gts = [[(0, 0, 10, 10), (50, 0, 10, 10)]]
    preds = [[(0, 0, 10, 10), (50, 0, 10, 10)]]
    ap = average_precision(preds, gts, iou_threshold=0.5)
    assert math.isclose(ap, 1.0, rel_tol=1e-6)


def test_average_precision_no_gt():
    assert average_precision([[]], [[]]) == 0.0


def test_evaluate_detections_perfect():
    gts = [[(0, 0, 10, 10)], [(0, 0, 10, 10), (20, 20, 10, 10)]]
    preds = [[(0, 0, 10, 10)], [(0, 0, 10, 10), (20, 20, 10, 10)]]
    m = evaluate_detections(preds, gts, iou_threshold=0.5)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    assert m.tp == 3 and m.fp == 0 and m.fn == 0
    assert math.isclose(m.mean_iou, 1.0)


def test_evaluate_detections_fp_and_fn():
    gts = [[(0, 0, 10, 10), (50, 50, 10, 10)]]
    preds = [[(0, 0, 10, 10), (200, 200, 5, 5)]]  # 1 correct, 1 FP; 1 GT missed
    m = evaluate_detections(preds, gts, iou_threshold=0.5)
    assert m.tp == 1 and m.fp == 1 and m.fn == 1


def test_evaluate_detections_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_detections([[]], [[], []])


def test_metrics_summary_and_dict():
    m = evaluate_detections([[(0, 0, 10, 10)]], [[(0, 0, 10, 10)]])
    assert "P=1.000" in m.summary()
    assert m.as_dict()["f1"] == 1.0
