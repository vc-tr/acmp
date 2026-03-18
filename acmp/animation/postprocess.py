"""Post-processing for AI-generated animation frames.

Fixes color drift and blur from the diffusion model by matching
color histograms to the source panel and applying sharpening.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter


def match_color_histogram(source: Image.Image, target: Image.Image) -> Image.Image:
    """Transfer color distribution from source panel onto AI-generated frame.

    Matches mean and std of each LAB channel so the AI output preserves
    the original panel's colors while keeping AI-generated motion.
    """
    src = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = cv2.cvtColor(np.array(target.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)

    for ch in range(3):
        s_mean, s_std = src[:, :, ch].mean(), src[:, :, ch].std() + 1e-6
        t_mean, t_std = tgt[:, :, ch].mean(), tgt[:, :, ch].std() + 1e-6
        tgt[:, :, ch] = (tgt[:, :, ch] - t_mean) / t_std * s_std + s_mean

    tgt = np.clip(tgt, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(tgt, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def sharpen_frame(frame: Image.Image, amount: float = 0.5) -> Image.Image:
    """Apply unsharp mask sharpening to reduce upscale blur."""
    return frame.filter(ImageFilter.UnsharpMask(
        radius=2,
        percent=int(amount * 150),
        threshold=3,
    ))


def upscale_two_stage(
    frame: Image.Image,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """Two-stage upscale: 2x intermediate → sharpen → final size.

    Produces cleaner results than a single large upscale jump.
    """
    w, h = frame.size
    if w >= target_width and h >= target_height:
        return frame.resize((target_width, target_height), Image.LANCZOS)

    # Intermediate: 2x or halfway to target, whichever is smaller
    mid_w = min(w * 2, target_width)
    mid_h = min(h * 2, target_height)

    intermediate = frame.resize((mid_w, mid_h), Image.LANCZOS)
    intermediate = sharpen_frame(intermediate, amount=0.4)

    if mid_w == target_width and mid_h == target_height:
        return intermediate

    return intermediate.resize((target_width, target_height), Image.LANCZOS)
