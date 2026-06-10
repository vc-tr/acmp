"""Object-detection metrics for panel detection.

Implements the standard toolkit used to evaluate detectors:

  * IoU between boxes,
  * greedy IoU matching of predictions to ground truth,
  * precision / recall / F1 at a fixed IoU threshold,
  * Average Precision (AP@IoU) via the all-point PR-curve integral (COCO-style),
  * dataset-level aggregation.

All boxes are ``(x, y, w, h)`` in pixels, matching the rest of acmp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    """Intersection-over-Union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(
    preds: list[Box],
    gts: list[Box],
    iou_threshold: float = 0.5,
    scores: list[float] | None = None,
) -> tuple[list[int], list[float]]:
    """Greedily match predictions to ground-truth boxes by descending score.

    Each GT can be matched at most once (the standard detection protocol).

    Args:
        preds: predicted boxes.
        gts: ground-truth boxes.
        iou_threshold: minimum IoU for a match to count as a true positive.
        scores: optional confidence per prediction; if None, preds are taken
            in the given order.

    Returns:
        (match_gt_index, match_iou) per prediction. ``match_gt_index[i]`` is the
        GT index matched by prediction i, or -1 if it is a false positive.
    """
    order = (
        sorted(range(len(preds)), key=lambda i: scores[i], reverse=True)
        if scores is not None
        else list(range(len(preds)))
    )

    matched_gt = [False] * len(gts)
    match_gt_index = [-1] * len(preds)
    match_iou = [0.0] * len(preds)

    for i in order:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched_gt[j]:
                continue
            iou = box_iou(preds[i], gt)
            if iou >= iou_threshold and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched_gt[best_j] = True
            match_gt_index[i] = best_j
            match_iou[i] = best_iou

    return match_gt_index, match_iou


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 from confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def average_precision(
    all_preds: list[list[Box]],
    all_gts: list[list[Box]],
    all_scores: list[list[float]] | None = None,
    iou_threshold: float = 0.5,
) -> float:
    """Average Precision at a single IoU threshold (all-point integration).

    Aggregates detections across every image, ranks them by confidence, sweeps
    the PR curve and integrates it — the COCO/Pascal-VOC style AP.

    If ``all_scores`` is None (e.g. the heuristic detector emits no confidence),
    box area is used as a proxy ranking so the metric is still defined; treat
    that AP as indicative rather than exact.
    """
    total_gt = sum(len(g) for g in all_gts)
    if total_gt == 0:
        return 0.0

    records: list[tuple[float, bool]] = []  # (score, is_true_positive)
    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        if all_scores is not None:
            scores = all_scores[img_idx]
        else:
            scores = [w * h for (_, _, w, h) in preds]  # area proxy
        match_gt_index, _ = match_boxes(preds, gts, iou_threshold, scores)
        for i in range(len(preds)):
            records.append((scores[i], match_gt_index[i] >= 0))

    if not records:
        return 0.0

    records.sort(key=lambda r: r[0], reverse=True)
    tp_cum = np.cumsum([1 if is_tp else 0 for _, is_tp in records])
    fp_cum = np.cumsum([0 if is_tp else 1 for _, is_tp in records])

    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # All-point interpolation: integrate precision over recall.
    recalls = np.concatenate(([0.0], recalls, [recalls[-1]]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    # Make precision monotonically decreasing (envelope).
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))
    return ap


@dataclass
class DetectionMetrics:
    """Aggregate detection metrics over a dataset."""

    precision: float
    recall: float
    f1: float
    ap50: float
    mean_iou: float  # mean IoU of matched (true-positive) predictions
    tp: int
    fp: int
    fn: int
    n_images: int
    iou_threshold: float = 0.5

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "ap50": round(self.ap50, 4),
            "mean_iou": round(self.mean_iou, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_images": self.n_images,
            "iou_threshold": self.iou_threshold,
        }

    def summary(self) -> str:
        return (
            f"images={self.n_images}  P={self.precision:.3f}  R={self.recall:.3f}  "
            f"F1={self.f1:.3f}  AP@{self.iou_threshold:g}={self.ap50:.3f}  "
            f"mIoU={self.mean_iou:.3f}  (TP={self.tp} FP={self.fp} FN={self.fn})"
        )


def evaluate_detections(
    all_preds: list[list[Box]],
    all_gts: list[list[Box]],
    all_scores: list[list[float]] | None = None,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Evaluate a detector's predictions against ground truth over a dataset.

    Args:
        all_preds: per-image list of predicted boxes.
        all_gts: per-image list of ground-truth boxes.
        all_scores: optional per-image confidence scores (for AP ranking).
        iou_threshold: IoU at which a prediction counts as a true positive.

    Returns:
        Aggregated :class:`DetectionMetrics`.
    """
    if len(all_preds) != len(all_gts):
        raise ValueError("all_preds and all_gts must have the same length")

    tp = fp = fn = 0
    matched_ious: list[float] = []

    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        scores = all_scores[img_idx] if all_scores is not None else None
        match_gt_index, match_iou = match_boxes(preds, gts, iou_threshold, scores)
        img_tp = sum(1 for m in match_gt_index if m >= 0)
        tp += img_tp
        fp += len(preds) - img_tp
        fn += len(gts) - img_tp
        matched_ious.extend(iou for k, iou in enumerate(match_iou)
                            if match_gt_index[k] >= 0)

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    ap50 = average_precision(all_preds, all_gts, all_scores, iou_threshold)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=ap50,
        mean_iou=mean_iou,
        tp=tp,
        fp=fp,
        fn=fn,
        n_images=len(all_preds),
        iou_threshold=iou_threshold,
    )
