"""Character/foreground segmentation using rembg.

Separates characters from backgrounds in manga/comic panels using the
isnet-anime model, which is optimized for anime/manga art styles.
~43MB model, runs on CPU to avoid competing with Wan VACE for GPU memory.
"""

from __future__ import annotations

import gc
import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded rembg session
_session = None


def _get_session():
    """Get or create the rembg session with isnet-anime model."""
    global _session
    if _session is None:
        from rembg import new_session
        logger.info("Loading isnet-anime segmentation model...")
        _session = new_session("isnet-anime")
        logger.info("Segmentation model loaded")
    return _session


def unload_session():
    """Unload the rembg session to free memory before loading Wan VACE."""
    global _session
    if _session is not None:
        del _session
        _session = None
        gc.collect()
        logger.info("Segmentation model unloaded")


def segment_character(
    panel: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Separate character(s) from background in a manga/comic panel.

    Args:
        panel: Source panel image (PIL RGB).

    Returns:
        Tuple of (character_rgba, background_rgb, alpha_mask):
          - character_rgba: Characters on transparent background (RGBA)
          - background_rgb: Reconstructed background with characters removed (RGB)
          - alpha_mask: Grayscale mask where white=character (L mode)
    """
    from rembg import remove

    session = _get_session()
    panel_rgb = panel.convert("RGB")

    # Remove background → RGBA with transparent background
    character_rgba = remove(panel_rgb, session=session)

    # Extract alpha channel as the segmentation mask
    alpha_mask = character_rgba.split()[-1]  # L mode, 0=bg, 255=character

    # Create background by inpainting the character region
    mask_np = np.array(alpha_mask)
    foreground_mask = mask_np > 128  # Boolean mask

    if foreground_mask.any():
        background_rgb = _inpaint_background(panel_rgb, foreground_mask)
    else:
        # No character detected, background is the whole panel
        background_rgb = panel_rgb.copy()

    return character_rgba, background_rgb, alpha_mask


def _inpaint_background(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Inpaint the character region to reconstruct background."""
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Dilate mask for better inpainting coverage
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=3)

    inpainted = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)


def segment_panels(
    panels: list[Image.Image],
) -> list[tuple[Image.Image, Image.Image, Image.Image] | None]:
    """Segment characters from backgrounds for all panels.

    Processes all panels at once, then unloads the model to free memory
    before Wan VACE animation begins.

    Returns:
        List of (character_rgba, background_rgb, alpha_mask) tuples,
        or None for panels where segmentation failed.
    """
    results = []

    for i, panel in enumerate(panels):
        try:
            logger.info(f"Segmenting panel {i+1}/{len(panels)}...")
            result = segment_character(panel)
            results.append(result)
        except Exception as e:
            logger.warning(f"Segmentation failed for panel {i+1}: {e}")
            results.append(None)

    # Unload model to free memory for animation
    unload_session()

    return results
