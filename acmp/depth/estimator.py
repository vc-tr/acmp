"""Depth estimation for parallax animation.

Uses MiDaS for Apple Silicon MPS compatibility, with Depth Anything V2 as an option.
"""

from __future__ import annotations

import logging
import numpy as np
from PIL import Image

from acmp.config import DepthConfig

logger = logging.getLogger(__name__)

# Lazy-loaded model instance
_model = None
_transform = None
_device = None


def _get_device(config: DepthConfig) -> str:
    """Determine the best available device."""
    if config.device != "auto":
        return config.device

    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, AttributeError):
        pass
    return "cpu"


def _load_model(config: DepthConfig):
    """Load the depth estimation model."""
    global _model, _transform, _device

    import torch

    _device = _get_device(config)
    logger.info(f"Loading depth model '{config.model}' on {_device}")

    if config.model.startswith("midas"):
        model_type = "DPT_Small" if "small" in config.model else "DPT_Large"
        _model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        _model.to(_device)
        _model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        _transform = (
            midas_transforms.small_transform
            if "small" in config.model
            else midas_transforms.dpt_transform
        )
    else:
        raise ValueError(f"Unknown depth model: {config.model}")

    logger.info(f"Depth model loaded successfully")


def estimate_depth(
    image: Image.Image,
    config: DepthConfig | None = None,
) -> np.ndarray:
    """Estimate a relative depth map from a single image.

    Args:
        image: PIL Image (RGB).
        config: Depth estimation config.

    Returns:
        Depth map as float32 numpy array, normalized to [0, 1].
        Higher values = farther from camera.
    """
    global _model, _transform, _device

    if config is None:
        config = DepthConfig()

    import torch

    if _model is None:
        _load_model(config)

    # Prepare input
    img_np = np.array(image)
    input_batch = _transform(img_np).to(_device)

    # Inference
    with torch.no_grad():
        prediction = _model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize to [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)

    return depth.astype(np.float32)


def depth_to_layers(
    depth_map: np.ndarray,
    num_layers: int = 3,
) -> list[np.ndarray]:
    """Split a depth map into discrete layer masks.

    Args:
        depth_map: Normalized depth map [0, 1].
        num_layers: Number of depth layers to create.

    Returns:
        List of binary masks (bool arrays), from foreground to background.
    """
    thresholds = np.linspace(0, 1, num_layers + 1)
    masks = []

    for i in range(num_layers):
        low = thresholds[i]
        high = thresholds[i + 1]
        mask = (depth_map >= low) & (depth_map < high)
        masks.append(mask)

    return masks
