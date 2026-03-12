"""Ken Burns effect - slow pan and zoom animations on static images."""

from __future__ import annotations

import numpy as np
from PIL import Image

from acmp.config import KenBurnsConfig


def render_ken_burns_frames(
    panel: Image.Image,
    num_frames: int,
    output_size: tuple[int, int],
    config: KenBurnsConfig | None = None,
    effect: str = "zoom_in",
) -> list[Image.Image]:
    """Render Ken Burns animation frames (zoom + pan).

    Args:
        panel: Source panel image (PIL RGB).
        num_frames: Total frames to render.
        output_size: (width, height) of output.
        config: Ken Burns config.
        effect: 'zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'pan_up', 'pan_down'.

    Returns:
        List of rendered PIL Images (RGB).
    """
    if config is None:
        config = KenBurnsConfig()

    out_w, out_h = output_size
    zoom_start, zoom_end = config.zoom_range

    frames = []

    for frame_idx in range(num_frames):
        t = frame_idx / max(num_frames - 1, 1)  # 0.0 to 1.0
        # Ease in-out for smooth motion
        t_eased = _ease_in_out(t)

        if effect == "zoom_in":
            frame = _zoom_frame(panel, out_w, out_h, zoom_start, zoom_end, t_eased, 0.5, 0.5)
        elif effect == "zoom_out":
            frame = _zoom_frame(panel, out_w, out_h, zoom_end, zoom_start, t_eased, 0.5, 0.5)
        elif effect == "pan_left":
            frame = _pan_frame(panel, out_w, out_h, t_eased, direction="left")
        elif effect == "pan_right":
            frame = _pan_frame(panel, out_w, out_h, t_eased, direction="right")
        elif effect == "pan_up":
            frame = _pan_frame(panel, out_w, out_h, t_eased, direction="up")
        elif effect == "pan_down":
            frame = _pan_frame(panel, out_w, out_h, t_eased, direction="down")
        else:
            frame = _zoom_frame(panel, out_w, out_h, zoom_start, zoom_end, t_eased, 0.5, 0.5)

        frames.append(frame)

    return frames


def _zoom_frame(
    image: Image.Image,
    out_w: int,
    out_h: int,
    zoom_start: float,
    zoom_end: float,
    t: float,
    focus_x: float,
    focus_y: float,
) -> Image.Image:
    """Render a single frame with zoom effect centered on focus point."""
    img_w, img_h = image.size
    zoom = zoom_start + (zoom_end - zoom_start) * t

    # Size of the crop window (smaller zoom = wider crop, larger zoom = tighter crop)
    crop_w = out_w / zoom
    crop_h = out_h / zoom

    # Scale source image to be large enough
    scale = max(out_w / img_w, out_h / img_h) * max(zoom_start, zoom_end) * 1.1
    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)
    scaled = image.resize((scaled_w, scaled_h), Image.LANCZOS)

    # Center point for crop
    cx = scaled_w * focus_x
    cy = scaled_h * focus_y

    # Crop region
    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x2 = int(cx + crop_w / 2)
    y2 = int(cy + crop_h / 2)

    # Clamp
    x1 = max(0, min(x1, scaled_w - int(crop_w)))
    y1 = max(0, min(y1, scaled_h - int(crop_h)))
    x2 = x1 + int(crop_w)
    y2 = y1 + int(crop_h)

    cropped = scaled.crop((x1, y1, x2, y2))
    return cropped.resize((out_w, out_h), Image.LANCZOS)


def _pan_frame(
    image: Image.Image,
    out_w: int,
    out_h: int,
    t: float,
    direction: str = "left",
) -> Image.Image:
    """Render a single frame with pan effect."""
    img_w, img_h = image.size

    # Scale image to fill output height/width with room for panning
    if direction in ("left", "right"):
        scale = out_h / img_h * 1.3  # Extra width for panning
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        scaled = image.resize((scaled_w, scaled_h), Image.LANCZOS)

        pan_range = scaled_w - out_w
        if direction == "left":
            x = int(pan_range * (1 - t))
        else:
            x = int(pan_range * t)
        y = (scaled_h - out_h) // 2

    else:  # up/down
        scale = out_w / img_w * 1.3
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        scaled = image.resize((scaled_w, scaled_h), Image.LANCZOS)

        pan_range = scaled_h - out_h
        x = (scaled_w - out_w) // 2
        if direction == "up":
            y = int(pan_range * (1 - t))
        else:
            y = int(pan_range * t)

    x = max(0, min(x, scaled_w - out_w))
    y = max(0, min(y, scaled_h - out_h))

    return scaled.crop((x, y, x + out_w, y + out_h))


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out curve (cubic)."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2
