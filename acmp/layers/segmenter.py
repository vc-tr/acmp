"""Foreground/background segmentation for layer separation.

Provides depth-based segmentation (MVP) and optional SAM2-based segmentation.
"""

from __future__ import annotations

import logging
import numpy as np
from PIL import Image

from acmp.depth.estimator import estimate_depth, depth_to_layers
from acmp.config import LayerConfig, DepthConfig

logger = logging.getLogger(__name__)


def segment_layers(
    panel: Image.Image,
    layer_config: LayerConfig | None = None,
    depth_config: DepthConfig | None = None,
) -> list[tuple[Image.Image, np.ndarray]]:
    """Segment a panel image into depth layers.

    Args:
        panel: Panel image (PIL RGB).
        layer_config: Layer separation config.
        depth_config: Depth estimation config.

    Returns:
        List of (layer_image, mask) tuples, from foreground to background.
        layer_image is RGBA with transparency where the layer doesn't exist.
        mask is a boolean numpy array.
    """
    if layer_config is None:
        layer_config = LayerConfig()
    if depth_config is None:
        depth_config = DepthConfig()

    # Get depth map
    depth_map = estimate_depth(panel, depth_config)

    # Split into layer masks
    masks = depth_to_layers(depth_map, layer_config.num_layers)

    # Create layer images with alpha
    panel_arr = np.array(panel)
    layers = []

    for mask in masks:
        # Create RGBA image
        rgba = np.zeros((*panel_arr.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = panel_arr
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)

        layer_img = Image.fromarray(rgba, "RGBA")
        layers.append((layer_img, mask))

    logger.info(f"Segmented panel into {len(layers)} layers")
    return layers
