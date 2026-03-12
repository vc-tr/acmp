#!/usr/bin/env python3
"""Download model weights for ACMP v2 pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def download_wan_vace():
    """Download Wan 2.1 VACE 1.3B model via diffusers."""
    print("\n--- Wan 2.1 VACE 1.3B (Image-to-Video) ---")
    print("Model size: ~3GB download")
    print("This is the core AI animation model.\n")

    try:
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        print(f"Downloading from {model_id}...")

        print("  Downloading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32,
        )
        del vae

        print("  Downloading pipeline (transformer, text encoder, etc.)...")
        pipe = WanVACEPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32,
        )
        del pipe

        print("Wan VACE 1.3B downloaded successfully!")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install diffusers accelerate torch")
    except Exception as e:
        print(f"Download failed: {e}")


def download_midas():
    """Download MiDaS depth estimation model (for parallax fallback)."""
    print("\n--- MiDaS DPT-Small (Depth Estimation) ---")
    print("Used for parallax fallback animation.\n")

    try:
        import torch
        print("Downloading via torch hub...")
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Small", trust_repo=True)
        del model
        print("MiDaS DPT-Small downloaded successfully!")
    except ImportError:
        print("torch not installed. Skipping.")
    except Exception as e:
        print(f"Download failed: {e}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("ACMP v2 Model Downloader")
    print("=" * 50)

    if "--wan" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        download_wan_vace()

    if "--midas" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        download_midas()

    print("\n" + "=" * 50)
    print("Done! Models cached by HuggingFace/torch hub.")
    print("  acmp process --input <chapter> --output video.mp4")
    print("=" * 50)


if __name__ == "__main__":
    main()
