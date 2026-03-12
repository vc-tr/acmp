"""Image utility functions."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image file as a PIL Image (RGB)."""
    return Image.open(path).convert("RGB")


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to OpenCV BGR array."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def resize_to_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize image to fit within target dimensions, maintaining aspect ratio.

    The image is scaled so the largest dimension fits, then padded with black.
    """
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def crop_panel(img: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop a panel from an image given (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    return img.crop((x, y, x + w, y + h))


def is_image_file(path: Path) -> bool:
    """Check if a file is a supported image format."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_color_image(img: Image.Image, threshold: float = 0.1) -> bool:
    """Detect if an image is color (True) or grayscale/B&W (False).

    Uses saturation channel variance as the heuristic.
    """
    arr = np.array(img)
    if len(arr.shape) < 3:
        return False
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    return float(np.mean(saturation)) > (255 * threshold)
