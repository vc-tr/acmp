"""Main ACMP pipeline - orchestrates the full processing flow."""

from __future__ import annotations

import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from acmp.config import PipelineConfig
from acmp.ingest.loader import load_chapter
from acmp.panels.detector import detect_panels, detect_panels_vertical_scroll
from acmp.utils.reading_order import detect_reading_order, sort_panels_by_reading_order
from acmp.utils.image import crop_panel, resize_to_fit
from acmp.animation.engine import (
    select_animation_type,
    animate_panel,
    assemble_panel_animations,
)
from acmp.video.assembler import frames_to_video

logger = logging.getLogger(__name__)


def process_chapter(
    input_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig | None = None,
    use_depth: bool = False,
) -> Path:
    """Process a comic/manga chapter into an animated video.

    Args:
        input_path: Path to input (directory of images or PDF).
        output_path: Path for output MP4 video.
        config: Pipeline configuration. Uses defaults if None.
        use_depth: If True, use depth estimation for parallax layers.

    Returns:
        Path to the generated video.
    """
    if config is None:
        config = PipelineConfig.load()

    output_size = tuple(config.output.resolution)  # (width, height)
    fps = config.output.fps

    # Step 1: Load chapter pages
    logger.info("=" * 60)
    logger.info("ACMP Pipeline - Starting")
    logger.info("=" * 60)

    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Resolution: {output_size[0]}x{output_size[1]} @ {fps}fps")

    pages = load_chapter(input_path, dpi=config.input.dpi)
    logger.info(f"Loaded {len(pages)} pages")

    # Step 2: Detect reading order
    if config.input.reading_order == "auto":
        reading_order = detect_reading_order(pages)
    else:
        reading_order = config.input.reading_order
    logger.info(f"Reading order: {reading_order}")

    # Step 3: Detect panels on each page
    logger.info("Detecting panels...")
    all_panels: list[tuple[Image.Image, tuple[int, int, int, int], tuple[int, int]]] = []

    for page_idx, page in enumerate(tqdm(pages, desc="Panel detection", unit="page")):
        page_size = (page.width, page.height)

        if reading_order == "vertical":
            bboxes = detect_panels_vertical_scroll(page, config.panels)
        else:
            bboxes = detect_panels(page, config.panels)

        # Sort panels by reading order
        bboxes = sort_panels_by_reading_order(bboxes, reading_order, page.height)

        for bbox in bboxes:
            panel_img = crop_panel(page, bbox)
            all_panels.append((panel_img, bbox, page_size))

        logger.debug(f"  Page {page_idx + 1}: {len(bboxes)} panels")

    logger.info(f"Total panels detected: {len(all_panels)}")

    if not all_panels:
        # Fallback: treat each page as a single panel
        logger.warning("No panels detected, treating each page as a single panel")
        for page in pages:
            bbox = (0, 0, page.width, page.height)
            all_panels.append((page, bbox, (page.width, page.height)))

    # Step 4: Animate each panel
    logger.info("Animating panels...")
    panel_animations: list[list[Image.Image]] = []

    for i, (panel_img, bbox, page_size) in enumerate(
        tqdm(all_panels, desc="Animation", unit="panel")
    ):
        animation_type = select_animation_type(panel_img, bbox, page_size)

        # Optional: depth-based layer separation
        layers = None
        if use_depth and animation_type == "parallax":
            try:
                from acmp.layers.segmenter import segment_layers
                layers = segment_layers(panel_img)
            except Exception as e:
                logger.warning(f"  Depth estimation failed for panel {i+1}, using Ken Burns: {e}")
                animation_type = "ken_burns_zoom"

        frames = animate_panel(
            panel=panel_img,
            animation_type=animation_type,
            output_size=output_size,
            config=config.animation,
            layers=layers,
        )
        panel_animations.append(frames)
        logger.debug(f"  Panel {i+1}/{len(all_panels)}: {animation_type} → {len(frames)} frames")

    # Step 5: Assemble with transitions
    logger.info("Assembling video with transitions...")
    all_frames = assemble_panel_animations(panel_animations, config.animation, fps)

    # Step 6: Encode to video
    logger.info("Encoding video...")
    result_path = frames_to_video(
        frames=all_frames,
        output_path=output_path,
        fps=fps,
        codec=config.output.codec,
        bitrate=config.output.bitrate,
    )

    duration = len(all_frames) / fps
    logger.info("=" * 60)
    logger.info(f"Done! Video: {result_path}")
    logger.info(f"Duration: {duration:.1f}s | Panels: {len(all_panels)} | Frames: {len(all_frames)}")
    logger.info("=" * 60)

    return result_path
