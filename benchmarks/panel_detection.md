# Panel Detection Benchmark — heuristic (OpenCV) vs. learned (YOLOv8n)

**Reproduce:**

```bash
pip install -e ".[train]"
acmp train-detector --device cpu        # generate data → fine-tune YOLOv8n → benchmark
acmp eval --weights runs/panel_detector/train/panel_detector/weights/best.pt
```

YOLOv8n fine-tuned for 50 epochs (early-stopped at 42), `imgsz=448`, on 64 synthetic
train pages (50% degraded). Both detectors are evaluated on 16 held-out pages **per
condition** with the same IoU-matched metrics (`acmp/eval/metrics.py`) at IoU = 0.5.

| Condition | Detector | Precision | Recall | F1 | AP@0.5 | mIoU |
|---|---|---|---|---|---|---|
| clean | heuristic (OpenCV) | 1.000 | 1.000 | 1.000 | 1.000 | 0.906 |
| clean | learned (YOLOv8n) | 1.000 | 1.000 | 1.000 | 1.000 | **0.987** |
| degraded | heuristic (OpenCV) | 0.828 | 0.658 | 0.733 | 0.544 | 0.792 |
| degraded | learned (YOLOv8n) | **1.000** | **1.000** | **1.000** | **1.000** | **0.987** |

## Reading the results

- On **clean** pages both detectors find every panel (F1 = AP = 1.0); the learned
  model localizes tighter (mIoU 0.987 vs 0.906).
- On **degraded** pages (Gaussian noise + blur + JPEG artefacts) the OpenCV
  heuristic drops sharply — recall 0.66, F1 0.73, AP@0.5 0.54 — because its
  adaptive-threshold/contour logic encodes clean-scan assumptions. The learned
  detector is essentially unaffected (F1 1.0).
- This is the textbook motivation for a learned detector: the heuristic is fast,
  free, and excellent on clean input, but brittle; the CNN generalises across
  appearance and degrades gracefully.

## Notes

- The heuristic emits no confidence, so its AP uses box area as a proxy ranking —
  treat its AP as indicative; F1/recall are the fair cross-detector comparison.
- Trained on **CPU**: ultralytics YOLO training is unstable on Apple MPS (silent
  non-convergence); MPS is fine for inference.
- Synthetic data keeps this fully reproducible without a license-restricted comic
  dataset; the same harness accepts real labeled pages.
