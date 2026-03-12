#!/usr/bin/env python3
"""Download model weights for ACMP pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "models"


def download_midas():
    """Download MiDaS depth estimation model via torch hub."""
    print("Downloading MiDaS model via torch hub...")
    print("(This will be cached automatically by torch hub)")

    import torch
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Small", trust_repo=True)
    print("MiDaS DPT_Small downloaded successfully!")
    del model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("ACMP Model Downloader")
    print("=" * 40)

    try:
        download_midas()
    except ImportError:
        print("torch not installed. Install with: pip install torch")
        print("Skipping model download (you can run without depth estimation)")

    print("\nDone! Models are cached by torch hub.")
    print("You can now run: acmp process --input <chapter> --output video.mp4")


if __name__ == "__main__":
    main()
