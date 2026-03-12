"""Wan 2.1 VACE 1.3B image-to-video animation.

Uses the Wan VACE pipeline (diffusers) to generate short animated clips
from static comic panel images. Optimized for 8GB Apple Silicon.
"""

from __future__ import annotations

import gc
import os
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

# Model ID on HuggingFace
WAN_VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

# Global pipeline (lazy loaded, explicitly freed between panels)
_pipeline = None


def _set_memory_env():
    """Set environment variables for 8GB Apple Silicon."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def _get_device() -> str:
    """Determine best available device."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(device: str | None = None):
    """Load the Wan VACE pipeline.

    Uses CPU offloading to fit in 8GB unified memory.
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    _set_memory_env()

    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    if device is None:
        device = _get_device()

    logger.info(f"Loading Wan VACE 1.3B on {device}...")
    logger.info("This may take a few minutes on first run (downloading ~3GB model)...")

    # Use float32 for MPS compatibility (bfloat16 not supported on MPS)
    dtype = torch.float32 if device == "mps" else torch.bfloat16

    vae = AutoencoderKLWan.from_pretrained(
        WAN_VACE_MODEL_ID,
        subfolder="vae",
        torch_dtype=dtype,
    )

    pipe = WanVACEPipeline.from_pretrained(
        WAN_VACE_MODEL_ID,
        vae=vae,
        torch_dtype=dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=3.0,  # Lower shift for lower resolution
    )

    # Critical for 8GB: offload model parts to CPU when not in use
    pipe.enable_model_cpu_offload()

    _pipeline = pipe
    logger.info("Wan VACE pipeline loaded successfully")
    return pipe


def unload_pipeline():
    """Explicitly unload the pipeline to free memory."""
    global _pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        gc.collect()

        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Wan VACE pipeline unloaded, memory freed")


def _prepare_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize and pad image to target dimensions (must be multiples of 16)."""
    # Round to multiples of 16
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16

    # Resize maintaining aspect ratio, then pad
    w, h = image.size
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    # Center on black canvas
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_w) // 2
    paste_y = (target_height - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def _create_vace_inputs(
    image: Image.Image,
    num_frames: int,
    width: int,
    height: int,
):
    """Create VACE-compatible conditioning inputs (first frame + mask).

    For image-to-video, we provide the first frame as reference
    and let the model generate the subsequent frames.
    """
    import torch
    import numpy as np

    # Prepare the reference image as the first frame
    ref_image = _prepare_image(image, width, height)

    # Create condition frames: first frame is the reference, rest are blank
    condition_frames = []
    condition_frames.append(ref_image)
    for _ in range(num_frames - 1):
        condition_frames.append(Image.new("RGB", (width, height), (0, 0, 0)))

    # Create masks: 0 = keep (known frame), 1 = generate
    # First frame mask = 0 (keep), rest = 1 (generate)
    masks = []
    masks.append(Image.new("L", (width, height), 0))  # Keep first frame
    for _ in range(num_frames - 1):
        masks.append(Image.new("L", (width, height), 255))  # Generate these

    return condition_frames, masks


def animate_panel(
    panel: Image.Image,
    motion_prompt: str = "subtle motion, gentle movement",
    num_frames: int = 25,
    width: int = 320,
    height: int = 576,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    device: str | None = None,
) -> list[Image.Image]:
    """Generate animated frames from a static panel image.

    Args:
        panel: Source panel image (PIL RGB).
        motion_prompt: Description of desired motion (from scene analyzer).
        num_frames: Number of frames to generate (must be 4k+1, e.g., 25).
        width: Output width (multiple of 16). Start low (320) for 8GB.
        height: Output height (multiple of 16). 576 for 9:16.
        num_inference_steps: Diffusion steps (lower = faster, less quality).
        guidance_scale: Prompt adherence strength.
        device: Compute device override.

    Returns:
        List of PIL Images (animation frames).
    """
    import torch

    pipe = load_pipeline(device)

    # Ensure num_frames follows 4k+1 rule
    k = max(1, (num_frames - 1) // 4)
    num_frames = 4 * k + 1

    # Ensure dimensions are multiples of 16
    width = (width // 16) * 16
    height = (height // 16) * 16

    logger.info(
        f"Generating {num_frames} frames at {width}x{height} "
        f"({num_inference_steps} steps, guidance={guidance_scale})"
    )
    logger.info(f"Motion prompt: {motion_prompt[:80]}...")

    # Create VACE inputs
    condition_frames, masks = _create_vace_inputs(panel, num_frames, width, height)

    try:
        with torch.no_grad():
            output = pipe(
                prompt=motion_prompt,
                negative_prompt="static, frozen, no motion, blurry, distorted, deformed",
                vace_frames=condition_frames,
                vace_masks=masks,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        frames = output.frames[0]  # List of PIL Images
        logger.info(f"Generated {len(frames)} animation frames")
        return frames

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            logger.error(
                f"OOM error at {width}x{height}. "
                f"Try reducing resolution or num_frames. Error: {e}"
            )
            # Clean up and re-raise
            gc.collect()
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        raise


def animate_panel_safe(
    panel: Image.Image,
    motion_prompt: str = "subtle motion, gentle movement",
    max_frames: int = 25,
    device: str | None = None,
) -> list[Image.Image] | None:
    """Animate a panel with automatic resolution fallback.

    Tries progressively lower resolutions if OOM errors occur.
    Returns None if all attempts fail.
    """
    # Resolution tiers to try (9:16 aspect ratio, multiples of 16)
    resolutions = [
        (480, 848),   # 480p — might work on 8GB with offloading
        (320, 576),   # 320p — most likely to work on 8GB
        (256, 448),   # 256p — last resort
    ]

    for width, height in resolutions:
        try:
            logger.info(f"Attempting animation at {width}x{height}...")
            frames = animate_panel(
                panel=panel,
                motion_prompt=motion_prompt,
                num_frames=max_frames,
                width=width,
                height=height,
                device=device,
            )
            return frames
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                logger.warning(f"OOM at {width}x{height}, trying lower resolution...")
                gc.collect()
                continue
            else:
                logger.error(f"Unexpected error: {e}")
                return None

    logger.error("All resolution attempts failed (OOM). Panel will use fallback animation.")
    return None
