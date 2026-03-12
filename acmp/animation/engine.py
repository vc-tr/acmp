"""Animation engine - orchestrates parallax, Ken Burns, and transitions."""

from __future__ import annotations

import logging
from PIL import Image

from acmp.config import AnimationConfig
from acmp.animation.parallax import render_parallax_frames
from acmp.animation.ken_burns import render_ken_burns_frames
from acmp.animation.transitions import crossfade

logger = logging.getLogger(__name__)


def select_animation_type(
    panel: Image.Image,
    panel_bbox: tuple[int, int, int, int],
    page_size: tuple[int, int],
) -> str:
    """Heuristically select the best animation type for a panel.

    Args:
        panel: Cropped panel image.
        panel_bbox: (x, y, w, h) of the panel on the page.
        page_size: (width, height) of the full page.

    Returns:
        Animation type: 'parallax', 'ken_burns_zoom', 'ken_burns_pan'.
    """
    _, _, pw, ph = panel_bbox
    page_w, page_h = page_size

    aspect = pw / max(ph, 1)
    panel_area_ratio = (pw * ph) / max(page_w * page_h, 1)

    # Wide panels → pan
    if aspect > 1.8:
        return "ken_burns_pan"

    # Large panels (splash pages, action scenes) → zoom
    if panel_area_ratio > 0.5:
        return "ken_burns_zoom"

    # Default: parallax (works well for character close-ups and standard panels)
    return "parallax"


def animate_panel(
    panel: Image.Image,
    animation_type: str,
    output_size: tuple[int, int],
    config: AnimationConfig,
    layers: list | None = None,
) -> list[Image.Image]:
    """Animate a single panel.

    Args:
        panel: Cropped panel image (PIL RGB).
        animation_type: 'parallax', 'ken_burns_zoom', or 'ken_burns_pan'.
        output_size: (width, height) for output frames.
        config: Animation configuration.
        layers: Optional depth-separated layers for parallax.

    Returns:
        List of animation frames (PIL RGB).
    """
    num_frames = int(config.seconds_per_panel * 24)  # Assume 24fps for now

    if animation_type == "parallax" and layers and len(layers) > 1:
        logger.debug(f"  Rendering parallax ({num_frames} frames, {len(layers)} layers)")
        return render_parallax_frames(
            layers=layers,
            num_frames=num_frames,
            output_size=output_size,
            config=config.parallax,
        )

    elif animation_type == "ken_burns_pan":
        # Determine pan direction based on panel aspect
        pw, ph = panel.size
        if pw > ph:
            effect = "pan_right"
        else:
            effect = "pan_down"

        logger.debug(f"  Rendering Ken Burns {effect} ({num_frames} frames)")
        return render_ken_burns_frames(
            panel=panel,
            num_frames=num_frames,
            output_size=output_size,
            config=config.ken_burns,
            effect=effect,
        )

    else:
        # Default: Ken Burns zoom in
        logger.debug(f"  Rendering Ken Burns zoom_in ({num_frames} frames)")
        return render_ken_burns_frames(
            panel=panel,
            num_frames=num_frames,
            output_size=output_size,
            config=config.ken_burns,
            effect="zoom_in",
        )


def assemble_panel_animations(
    panel_frame_lists: list[list[Image.Image]],
    config: AnimationConfig,
    fps: int = 24,
) -> list[Image.Image]:
    """Combine individual panel animations with transitions.

    Args:
        panel_frame_lists: List of frame lists, one per panel.
        config: Animation config (for transition duration).
        fps: Frames per second.

    Returns:
        Single combined list of all frames with transitions.
    """
    if not panel_frame_lists:
        return []

    transition_frames = int(config.transition_duration * fps)
    combined = panel_frame_lists[0]

    for i in range(1, len(panel_frame_lists)):
        combined = crossfade(combined, panel_frame_lists[i], transition_frames)
        logger.debug(f"  Added transition between panel {i} and {i+1}")

    logger.info(f"Assembled {len(combined)} total frames from {len(panel_frame_lists)} panels")
    return combined
