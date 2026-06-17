"""Train a YOLOv8 panel detector on synthetic data and benchmark it.

End-to-end ML workflow in one place:

  1. generate a labeled train/val/test split (clean + degraded pages),
  2. export it to YOLO format,
  3. fine-tune a YOLOv8n detector,
  4. benchmark the trained model against the OpenCV heuristic on held-out
     clean and degraded test sets, and
  5. write a Markdown results report.

Kept small enough to run on CPU/MPS without a GPU; scale up epochs / dataset
size / image size for stronger numbers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from acmp.eval.dataset import write_yolo_dataset
from acmp.eval.runner import benchmark_suite, format_results, heuristic_detector
from acmp.eval.synthetic import generate_split
from acmp.panels.yolo_detector import YoloPanelDetector, _resolve_device

logger = logging.getLogger(__name__)


def train_panel_detector(
    out_dir: str | Path = "runs/panel_detector",
    epochs: int = 30,
    imgsz: int = 512,
    batch: int = 8,
    n_train: int = 60,
    n_val: int = 16,
    n_test: int = 16,
    base_model: str = "yolov8n.pt",
    device: str = "auto",
    degrade_frac: float = 0.5,
    seed: int = 0,
) -> dict:
    """Train and benchmark a panel detector. Returns a results dict."""
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device)
    logger.info(f"Training on device={device}, {epochs} epochs, imgsz={imgsz}")

    # 1-2. Build dataset split and export to YOLO format.
    split = generate_split(
        n_train=n_train, n_val=n_val, n_test=n_test, degrade_frac=degrade_frac, seed=seed
    )
    dataset_dir = out_dir / "dataset"
    data_yaml = write_yolo_dataset(split["train"], split["val"], dataset_dir)
    logger.info(f"Wrote YOLO dataset to {dataset_dir}")

    # 3. Fine-tune YOLOv8n.
    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        # Absolute path so ultralytics doesn't nest it under its own runs_dir setting.
        project=str((out_dir / "train").resolve()),
        name="panel_detector",
        exist_ok=True,
        seed=seed,
        verbose=False,
        plots=False,
        workers=2,
        patience=15,  # early-stop when val mAP plateaus
        # Hyperparameters tuned for full-page panel layouts. Mosaic composites 4
        # pages into one — meaningless for page-level layout — so disable it and
        # keep only light, geometry-safe augmentation. (optimizer left on "auto".)
        mosaic=0.0,
        close_mosaic=0,
        scale=0.2,
        translate=0.05,
        erasing=0.0,
        degrees=0.0,
        perspective=0.0,
    )
    # Read the real save_dir from the trainer — robust to ultralytics path settings.
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    logger.info(f"Best weights: {best_weights}")

    # 4. Benchmark heuristic vs trained YOLO on held-out clean & degraded sets.
    yolo = YoloPanelDetector(best_weights, device=device, imgsz=imgsz)
    detectors = {"heuristic (OpenCV)": heuristic_detector, "learned (YOLOv8n)": yolo}
    conditions = {"clean": split["test_clean"], "degraded": split["test_degraded"]}
    results = benchmark_suite(detectors, conditions, iou_threshold=0.5)

    # 5. Report.
    table = format_results(results)
    report = (
        "# Panel Detection Benchmark\n\n"
        f"YOLOv8n fine-tuned for {epochs} epochs (imgsz={imgsz}) on "
        f"{n_train} synthetic train pages ({int(degrade_frac * 100)}% degraded), "
        f"evaluated on {n_test} held-out pages per condition.\n\n"
        f"{table}\n\n"
        f"Weights: `{best_weights}`\n"
    )
    (out_dir / "results.md").write_text(report)
    (out_dir / "results.json").write_text(
        json.dumps(
            {c: {n: m.as_dict() for n, m in d.items()} for c, d in results.items()},
            indent=2,
        )
    )
    logger.info(f"Wrote report to {out_dir / 'results.md'}")
    return {"weights": str(best_weights), "results": results, "report": report}
