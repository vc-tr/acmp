"""Background inpainting to reconstruct hidden regions behind foreground layers."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import logging

from acmp.utils.image import pil_to_cv2, cv2_to_pil

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
        method: 'opencv' for cv2.inpaint, 'lama' for LaMa (future).

    Returns:
        Inpainted background image (PIL RGB) with foreground regions filled.
    """
    if method == "opencv":
        return _inpaint_opencv(image, foreground_mask)
    elif method == "lama":
        raise NotImplementedError("LaMa inpainting not yet implemented. Use 'opencv'.")
    else:
        raise ValueError(f"Unknown inpainting method: {method}")


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
