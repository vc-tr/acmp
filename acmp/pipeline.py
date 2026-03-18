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
    quality: str = "fast",
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

    # Log system memory info (helpful for diagnosing OOM)
    try:
        import os
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        logger.info(f"System RAM: {mem_gb:.1f}GB")
        if mem_gb <= 8:
            logger.info("8GB detected: using reduced resolution/frames for AI animation")
    except Exception:
        pass

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

    # ========== Step 4b: Character segmentation ==========
    segmentation_data = None
    if use_ai and config.segmentation.enabled and config.ai_motion.character_aware:
        try:
            logger.info("Segmenting characters from backgrounds...")
            from acmp.layers.character_segmenter import segment_panels
            segmentation_data = segment_panels(panel_images)
            seg_count = sum(1 for s in segmentation_data if s is not None)
            logger.info(f"Segmented {seg_count}/{len(panel_images)} panels successfully")
        except ImportError:
            logger.warning("rembg not installed. Skipping character segmentation. Install with: pip install rembg[cpu]")
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}")

    # ========== Step 5: Animate panels ==========
    panel_animations: list[list[Image.Image]] = []

    if use_ai:
        logger.info("Animating panels with Wan VACE 1.3B (AI)...")
        panel_animations = _animate_with_ai(
            panel_images, analyses, output_size, config, fps, quality,
            segmentation_data=segmentation_data,
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


def _extend_frames_pingpong(
    frames: list[Image.Image],
    target_count: int,
    crossfade_length: int = 3,
) -> list[Image.Image]:
    """Extend frames to target count using ping-pong looping with crossfade.

    Instead of repeating [1,2,3,1,2,3,...] which creates a visible jump,
    this creates [1,2,3,2,1,2,3,...] — a natural back-and-forth oscillation.
    Crossfade blending at reversal points hides the direction change.
    """
    if len(frames) >= target_count:
        return frames[:target_count]

    if len(frames) < 2:
        return frames * target_count

    # Build one ping-pong cycle: forward + reverse (skip endpoints to avoid freeze)
    reverse_middle = list(reversed(frames[1:-1]))

    # Apply crossfade at the reversal point (end of forward → start of reverse)
    cf = min(crossfade_length, len(reverse_middle), len(frames) - 1)
    if cf > 0 and reverse_middle:
        for j in range(cf):
            alpha = (j + 1) / (cf + 1)
            # Blend the last cf frames of forward with first cf of reverse
            forward_frame = frames[-(cf - j)]
            reverse_frame = reverse_middle[j]
            reverse_middle[j] = Image.blend(forward_frame, reverse_frame, alpha)

    cycle = list(frames) + reverse_middle  # One full oscillation

    # Repeat cycles until we have enough frames
    result = []
    while len(result) < target_count:
        remaining = target_count - len(result)
        result.extend(cycle[:remaining])

    return result[:target_count]


def _animate_with_ai(
    panels: list[Image.Image],
    analyses: list,
    output_size: tuple[int, int],
    config: PipelineConfig,
    fps: int,
    quality: str = "fast",
    segmentation_data: list | None = None,
) -> list[list[Image.Image]]:
    """Animate panels using Wan VACE AI model with v1 fallback.

    When segmentation_data is available, uses composite animation:
    characters get AI animation, backgrounds get Ken Burns (preserves quality).
    """
    from acmp.animation.wan_animator import animate_panel_safe, unload_pipeline
    from acmp.animation.ken_burns import render_ken_burns_frames
    from acmp.animation.postprocess import match_color_histogram, sharpen_frame, upscale_two_stage

    panel_animations = []
    num_frames_per_panel = int(config.animation.seconds_per_panel * fps)
    out_w, out_h = output_size

    for i, (panel, analysis) in enumerate(
        tqdm(list(zip(panels, analyses)), desc="AI Animation", unit="panel")
    ):
        logger.info(f"Panel {i+1}/{len(panels)}: {analysis.action}")

        seg = segmentation_data[i] if segmentation_data and i < len(segmentation_data) else None

        if seg is not None:
            # Character-aware composite animation
            character_rgba, background_rgb, alpha_mask = seg
            frames = _animate_composite(
                panel, character_rgba, background_rgb, alpha_mask,
                analysis, num_frames_per_panel, output_size, quality,
            )
            if frames:
                panel_animations.append(frames)
                logger.info(f"  Composite animation: character AI + background Ken Burns")
                gc.collect()
                continue

        # Full-panel AI animation (no segmentation or composite failed)
        motion_prompt = analysis.character_motion_prompt or analysis.motion_prompt
        ai_frames = animate_panel_safe(
            panel=panel,
            motion_prompt=motion_prompt,
            max_frames=min(num_frames_per_panel, 17),
            quality=quality,
        )

        if ai_frames:
            source_panel = panel.convert("RGB")
            resized_frames = []
            for f in ai_frames:
                upscaled = upscale_two_stage(f, out_w, out_h)
                color_matched = match_color_histogram(source_panel, upscaled)
                sharpened = sharpen_frame(color_matched)
                resized_frames.append(sharpened)

            resized_frames = _extend_frames_pingpong(
                resized_frames, num_frames_per_panel
            )
            panel_animations.append(resized_frames[:num_frames_per_panel])
            logger.info(f"  AI animation: {len(ai_frames)} frames generated")
        else:
            logger.warning(f"  AI failed, falling back to Ken Burns for panel {i+1}")
            kb_frames = render_ken_burns_frames(
                panel=panel,
                num_frames=num_frames_per_panel,
                output_size=output_size,
                effect="zoom_in",
            )
            panel_animations.append(kb_frames)

        gc.collect()

    try:
        unload_pipeline()
    except Exception:
        pass

    return panel_animations


def _animate_composite(
    panel: Image.Image,
    character_rgba: Image.Image,
    background_rgb: Image.Image,
    alpha_mask: Image.Image,
    analysis,
    num_frames: int,
    output_size: tuple[int, int],
    quality: str,
) -> list[Image.Image] | None:
    """Composite animation: AI for characters, Ken Burns for background.

    Characters get Wan VACE animation (the part that benefits from AI motion).
    Background gets Ken Burns (preserves original quality, no AI artifacts).
    Results are alpha-composited together per frame.
    """
    import cv2
    import numpy as np
    from acmp.animation.wan_animator import animate_panel_safe
    from acmp.animation.ken_burns import render_ken_burns_frames
    from acmp.animation.postprocess import match_color_histogram, sharpen_frame, upscale_two_stage

    out_w, out_h = output_size

    # 1. Animate background with Ken Burns (clean, no AI blur)
    camera = analysis.camera_suggestion.lower()
    if "pan" in camera and "left" in camera:
        effect = "pan_left"
    elif "pan" in camera and "right" in camera:
        effect = "pan_right"
    elif "zoom" in camera and "out" in camera:
        effect = "zoom_out"
    else:
        effect = "zoom_in"

    bg_frames = render_ken_burns_frames(
        panel=background_rgb,
        num_frames=num_frames,
        output_size=output_size,
        effect=effect,
    )

    # 2. Animate character with Wan VACE
    char_prompt = analysis.character_motion_prompt or analysis.motion_prompt
    char_ai_frames = animate_panel_safe(
        panel=character_rgba.convert("RGB"),
        motion_prompt=char_prompt,
        max_frames=min(num_frames, 17),
        quality=quality,
    )

    if not char_ai_frames:
        return None

    # Post-process character frames
    source_char = character_rgba.convert("RGB")
    char_processed = []
    for f in char_ai_frames:
        upscaled = upscale_two_stage(f, out_w, out_h)
        color_matched = match_color_histogram(source_char, upscaled)
        sharpened = sharpen_frame(color_matched)
        char_processed.append(sharpened)

    char_processed = _extend_frames_pingpong(char_processed, num_frames)

    # 3. Prepare alpha mask for compositing
    mask_resized = alpha_mask.resize((out_w, out_h), Image.LANCZOS)
    # Soften edges: dilate slightly then blur for natural blending
    mask_np = np.array(mask_resized)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_np = cv2.dilate(mask_np, kernel, iterations=2)
    mask_np = cv2.GaussianBlur(mask_np, (7, 7), 0)
    soft_mask = Image.fromarray(mask_np).convert("L")

    # 4. Composite: character over background per frame
    composited = []
    for j in range(num_frames):
        bg = bg_frames[j] if j < len(bg_frames) else bg_frames[-1]
        char = char_processed[j] if j < len(char_processed) else char_processed[-1]
        # Use soft mask to blend character onto background
        composited.append(Image.composite(char, bg, soft_mask))

    return composited


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
