"""Main ACMP v2 pipeline - orchestrates the full processing flow.

v2 architecture:
  Input → Panel Detection → Scene Analysis (LLM) → AI Animation (Wan VACE)
  → Transitions → Video Assembly → Output MP4

Falls back to v1 Ken Burns/parallax when AI animation fails or is disabled.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from acmp.config import PipelineConfig
from acmp.ingest.loader import load_chapter
from acmp.panels.detector import detect_panels, detect_panels_vertical_scroll
from acmp.utils.reading_order import detect_reading_order, sort_panels_by_reading_order
from acmp.utils.image import crop_panel, resize_to_fit
from acmp.video.assembler import frames_to_video

logger = logging.getLogger(__name__)


def process_chapter(
    input_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig | None = None,
    use_ai: bool = True,
    use_depth: bool = False,
    llm_prefer: str = "claude",
    api_key: str | None = None,
) -> Path:
    """Process a comic/manga chapter into an animated video.

    Args:
        input_path: Path to input (directory of images or PDF).
        output_path: Path for output MP4 video.
        config: Pipeline configuration. Uses defaults if None.
        use_ai: If True, use Wan VACE AI animation (core v2 feature).
        use_depth: If True, use depth estimation for parallax (v1 fallback).
        llm_prefer: LLM for scene analysis ('claude', 'ollama', 'fallback').
        api_key: Anthropic API key for Claude (optional, reads env var if None).

    Returns:
        Path to the generated video.
    """
    if config is None:
        config = PipelineConfig.load()

    output_size = tuple(config.output.resolution)  # (width, height)
    fps = config.output.fps

    # ========== Step 1: Load chapter pages ==========
    logger.info("=" * 60)
    logger.info("ACMP v2 Pipeline - Starting")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Resolution: {output_size[0]}x{output_size[1]} @ {fps}fps")
    logger.info(f"AI animation: {'enabled' if use_ai else 'disabled (v1 fallback)'}")
    logger.info(f"Scene analysis: {llm_prefer}")

    pages = load_chapter(input_path, dpi=config.input.dpi)
    logger.info(f"Loaded {len(pages)} pages")

    # ========== Step 2: Detect reading order ==========
    if config.input.reading_order == "auto":
        reading_order = detect_reading_order(pages)
    else:
        reading_order = config.input.reading_order
    logger.info(f"Reading order: {reading_order}")

    # ========== Step 3: Detect panels ==========
    logger.info("Detecting panels...")
    all_panels: list[tuple[Image.Image, tuple[int, int, int, int], tuple[int, int]]] = []

    for page_idx, page in enumerate(tqdm(pages, desc="Panel detection", unit="page")):
        page_size = (page.width, page.height)

        if reading_order == "vertical":
            bboxes = detect_panels_vertical_scroll(page, config.panels)
        else:
            bboxes = detect_panels(page, config.panels)

        bboxes = sort_panels_by_reading_order(bboxes, reading_order, page.height)

        for bbox in bboxes:
            panel_img = crop_panel(page, bbox)
            all_panels.append((panel_img, bbox, page_size))

        logger.debug(f"  Page {page_idx + 1}: {len(bboxes)} panels")

    logger.info(f"Total panels detected: {len(all_panels)}")

    if not all_panels:
        logger.warning("No panels detected, treating each page as a single panel")
        for page in pages:
            bbox = (0, 0, page.width, page.height)
            all_panels.append((page, bbox, (page.width, page.height)))

    panel_images = [p[0] for p in all_panels]

    # ========== Step 4: Scene Analysis (LLM) ==========
    logger.info("Analyzing panels with LLM...")
    from acmp.scene.analyzer import analyze_chapter, PanelAnalysis

    analyses = analyze_chapter(
        panels=panel_images,
        prefer=llm_prefer,
        api_key=api_key,
    )

    for i, analysis in enumerate(analyses):
        logger.info(f"  Panel {i+1}: [{analysis.motion_intensity}] {analysis.description[:70]}")

    # ========== Step 5: Animate panels ==========
    panel_animations: list[list[Image.Image]] = []

    if use_ai:
        logger.info("Animating panels with Wan VACE 1.3B (AI)...")
        panel_animations = _animate_with_ai(
            panel_images, analyses, output_size, config, fps
        )
    else:
        logger.info("Animating panels with v1 engine (Ken Burns/parallax)...")
        panel_animations = _animate_with_v1(
            all_panels, analyses, output_size, config, use_depth
        )

    # ========== Step 6: Apply transitions ==========
    logger.info("Assembling with transitions...")
    all_frames = _apply_transitions(panel_animations, analyses, config, fps)

    # ========== Step 7: Encode video ==========
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


def _animate_with_ai(
    panels: list[Image.Image],
    analyses: list,
    output_size: tuple[int, int],
    config: PipelineConfig,
    fps: int,
) -> list[list[Image.Image]]:
    """Animate panels using Wan VACE AI model with v1 fallback."""
    from acmp.animation.wan_animator import animate_panel_safe, unload_pipeline
    from acmp.animation.ken_burns import render_ken_burns_frames

    panel_animations = []
    num_frames_per_panel = int(config.animation.seconds_per_panel * fps)

    for i, (panel, analysis) in enumerate(
        tqdm(list(zip(panels, analyses)), desc="AI Animation", unit="panel")
    ):
        logger.info(f"Panel {i+1}/{len(panels)}: {analysis.action}")

        # Try AI animation
        ai_frames = animate_panel_safe(
            panel=panel,
            motion_prompt=analysis.motion_prompt,
            max_frames=min(num_frames_per_panel, 25),  # Wan generates max ~25 frames
        )

        if ai_frames:
            # Resize AI frames to output resolution
            out_w, out_h = output_size
            resized_frames = [
                f.resize((out_w, out_h), Image.LANCZOS) for f in ai_frames
            ]

            # If AI generated fewer frames than needed, loop/extend
            while len(resized_frames) < num_frames_per_panel:
                resized_frames.extend(resized_frames[:num_frames_per_panel - len(resized_frames)])

            panel_animations.append(resized_frames[:num_frames_per_panel])
            logger.info(f"  AI animation: {len(ai_frames)} frames generated")
        else:
            # Fallback to Ken Burns
            logger.warning(f"  AI failed, falling back to Ken Burns for panel {i+1}")
            kb_frames = render_ken_burns_frames(
                panel=panel,
                num_frames=num_frames_per_panel,
                output_size=output_size,
                effect="zoom_in",
            )
            panel_animations.append(kb_frames)

        # Free memory between panels
        gc.collect()

    # Unload AI model after all panels processed
    try:
        unload_pipeline()
    except Exception:
        pass

    return panel_animations


def _animate_with_v1(
    all_panels: list[tuple[Image.Image, tuple[int, int, int, int], tuple[int, int]]],
    analyses: list,
    output_size: tuple[int, int],
    config: PipelineConfig,
    use_depth: bool,
) -> list[list[Image.Image]]:
    """Animate panels using v1 engine (Ken Burns/parallax)."""
    from acmp.animation.engine import select_animation_type, animate_panel

    panel_animations = []

    for i, ((panel_img, bbox, page_size), analysis) in enumerate(
        tqdm(list(zip(all_panels, analyses)), desc="V1 Animation", unit="panel")
    ):
        animation_type = select_animation_type(panel_img, bbox, page_size)

        # Use LLM suggestion to override animation type
        if analysis.motion_intensity == "high":
            animation_type = "ken_burns_zoom"
        elif "pan" in analysis.camera_suggestion.lower():
            animation_type = "ken_burns_pan"

        layers = None
        if use_depth and animation_type == "parallax":
            try:
                from acmp.layers.segmenter import segment_layers
                layers = segment_layers(panel_img)
            except Exception as e:
                logger.warning(f"  Depth failed for panel {i+1}: {e}")
                animation_type = "ken_burns_zoom"

        frames = animate_panel(
            panel=panel_img,
            animation_type=animation_type,
            output_size=output_size,
            config=config.animation,
            layers=layers,
        )
        panel_animations.append(frames)

    return panel_animations


def _apply_transitions(
    panel_animations: list[list[Image.Image]],
    analyses: list,
    config: PipelineConfig,
    fps: int,
) -> list[Image.Image]:
    """Apply context-aware transitions between panel animations."""
    from acmp.animation.transitions import crossfade, slide_transition

    if not panel_animations:
        return []

    transition_frames = int(config.animation.transition_duration * fps)
    combined = list(panel_animations[0])

    for i in range(1, len(panel_animations)):
        transition_type = "crossfade"
        if i < len(analyses):
            transition_type = analyses[i - 1].transition_to_next

        if transition_type in ("slide_left", "slide_right") and panel_animations[i]:
            # Use slide transition
            last_frame = combined[-1] if combined else panel_animations[i][0]
            first_frame = panel_animations[i][0]
            direction = "left" if transition_type == "slide_left" else "right"
            trans_frames = slide_transition(
                last_frame, first_frame, transition_frames, direction
            )
            combined.extend(trans_frames)
            combined.extend(panel_animations[i][1:])

        elif transition_type == "cut":
            # Hard cut, no transition
            combined.extend(panel_animations[i])

        elif transition_type == "fade_to_black":
            # Fade out to black, then fade in
            if combined:
                last = combined[-1]
                w, h = last.size
                black = Image.new("RGB", (w, h), (0, 0, 0))
                half = transition_frames // 2
                for j in range(half):
                    alpha = j / max(half - 1, 1)
                    combined.append(Image.blend(last, black, alpha))
                if panel_animations[i]:
                    first = panel_animations[i][0]
                    for j in range(half):
                        alpha = j / max(half - 1, 1)
                        combined.append(Image.blend(black, first, alpha))
                combined.extend(panel_animations[i])

        else:
            # Default: crossfade
            combined = crossfade(combined, panel_animations[i], transition_frames)

    return combined
