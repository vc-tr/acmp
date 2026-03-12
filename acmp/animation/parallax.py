"""Parallax animation - creates 2.5D depth effect from layered images."""

from __future__ import annotations

import numpy as np
from PIL import Image
import math

from acmp.config import ParallaxConfig


def render_parallax_frames(
    layers: list[tuple[Image.Image, np.ndarray]],
    num_frames: int,
    output_size: tuple[int, int],
    config: ParallaxConfig | None = None,
) -> list[Image.Image]:
    """Render parallax animation frames from depth-separated layers.

    Moves each layer at a different speed based on its depth position.
    Foreground layers (first in list) move more, background layers move less.

    Args:
        layers: List of (RGBA image, mask) from foreground to background.
        num_frames: Total number of frames to render.
        output_size: (width, height) of output frames.
        config: Parallax animation config.

    Returns:
        List of rendered PIL Images (RGB).
    """
    if config is None:
        config = ParallaxConfig()

    out_w, out_h = output_size
    num_layers = len(layers)
    frames = []

    for frame_idx in range(num_frames):
        t = frame_idx / max(num_frames - 1, 1)  # 0.0 to 1.0
        angle = t * 2 * math.pi * config.frequency

        # Create blank canvas
        canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))

        # Render layers back-to-front
        for layer_idx in reversed(range(num_layers)):
            layer_img, _ = layers[layer_idx]

            # Depth factor: background=0.0, foreground=1.0
            depth_factor = 1.0 - (layer_idx / max(num_layers - 1, 1))

            # Compute offset based on depth
            amplitude = config.amplitude * depth_factor

            if config.direction == "horizontal":
                dx = amplitude * math.sin(angle)
                dy = 0
            elif config.direction == "vertical":
                dx = 0
                dy = amplitude * math.sin(angle)
            elif config.direction == "circular":
                dx = amplitude * math.sin(angle)
                dy = amplitude * math.cos(angle) * 0.5
            else:
                dx = amplitude * math.sin(angle)
                dy = 0

            # Resize layer to fit output
            resized = _fit_layer(layer_img, out_w, out_h)

            # Apply offset
            offset_x = int(dx)
            offset_y = int(dy)

            # Composite layer onto canvas with alpha
            canvas = _composite_with_offset(canvas, resized, offset_x, offset_y)

        frames.append(canvas)

    return frames


def _fit_layer(layer: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize a layer to cover the target area (with slight overscan for parallax)."""
    # Add 10% overscan so parallax movement doesn't show edges
    overscan = 1.1
    w, h = layer.size

    scale = max(
        (target_w * overscan) / w,
        (target_h * overscan) / h,
    )
    new_w = int(w * scale)
    new_h = int(h * scale)

    return layer.resize((new_w, new_h), Image.LANCZOS)


def _composite_with_offset(
    canvas: Image.Image,
    layer: Image.Image,
    offset_x: int,
    offset_y: int,
) -> Image.Image:
    """Composite an RGBA layer onto an RGB canvas with offset."""
    cw, ch = canvas.size
    lw, lh = layer.size

    # Center the layer on canvas, then apply offset
    paste_x = (cw - lw) // 2 + offset_x
    paste_y = (ch - lh) // 2 + offset_y

    if layer.mode == "RGBA":
        # Create a temp canvas and paste with alpha
        temp = canvas.copy()
        temp.paste(layer, (paste_x, paste_y), layer)
        return temp
    else:
        canvas.paste(layer, (paste_x, paste_y))
        return canvas
