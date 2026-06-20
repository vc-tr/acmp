"""Background inpainting to reconstruct hidden regions behind foreground layers."""

from __future__ import annotations

import functools
import logging

import cv2
import numpy as np
from PIL import Image

from acmp.utils.image import cv2_to_pil, pil_to_cv2

logger = logging.getLogger(__name__)


def inpaint_background(
    image: Image.Image,
    foreground_mask: np.ndarray,
    method: str = "opencv",
) -> Image.Image:
    """Reconstruct the background behind foreground objects.

    Args:
        image: Original panel image (PIL RGB).
        foreground_mask: Boolean mask where True = foreground to remove.
        method: 'opencv' for cv2.inpaint, 'lama' for the LaMa model
            (falls back to OpenCV when LaMa's deps/weights are unavailable).

    Returns:
        Inpainted background image (PIL RGB) with foreground regions filled.
    """
    if method == "opencv":
        return _inpaint_opencv(image, foreground_mask)
    elif method == "lama":
        return _inpaint_lama(image, foreground_mask)
    else:
        raise ValueError(f"Unknown inpainting method: {method}")


@functools.lru_cache(maxsize=1)
def _load_lama():
    """Load and cache the LaMa model (downloads weights on first use)."""
    from simple_lama_inpainting import SimpleLama

    return SimpleLama()


def _inpaint_lama(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Inpaint with the LaMa model, degrading to OpenCV if it is unavailable.

    LaMa fills large masked regions far more cleanly than OpenCV's diffusion
    methods, but it pulls in torch and downloads model weights on first use.
    If the dependency or weights are unavailable we fall back to OpenCV so the
    pipeline keeps working everywhere.
    """
    try:
        lama = _load_lama()
        mask_img = Image.fromarray((mask.astype(np.uint8)) * 255)
        return lama(image.convert("RGB"), mask_img).convert("RGB")
    except Exception as exc:
        logger.warning("LaMa inpainting unavailable (%s); falling back to OpenCV.", exc)
        return _inpaint_opencv(image, mask)


def _inpaint_opencv(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Inpaint using OpenCV's Navier-Stokes or Telea method."""
    img_bgr = pil_to_cv2(image)

    # Dilate the mask slightly for better inpainting coverage
    mask_uint8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2)

    # Use Telea method (better for large regions) or NS method
    inpainted = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return cv2_to_pil(inpainted)
