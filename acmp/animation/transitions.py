"""Panel-to-panel transitions."""

from __future__ import annotations

import numpy as np
from PIL import Image


def crossfade(
    frames_a: list[Image.Image],
    frames_b: list[Image.Image],
    transition_frames: int,
) -> list[Image.Image]:
    """Create a crossfade transition between two panel animations.

    Blends the last N frames of animation A with the first N frames of animation B.

    Args:
        frames_a: Frames from the outgoing panel.
        frames_b: Frames from the incoming panel.
        transition_frames: Number of frames for the transition.

    Returns:
        Combined frame list with smooth crossfade.
    """
    if transition_frames <= 0:
        return frames_a + frames_b

    # Ensure we have enough frames
    transition_frames = min(transition_frames, len(frames_a), len(frames_b))

    result = list(frames_a[:-transition_frames])

    # Blend transition region
    for i in range(transition_frames):
        alpha = i / (transition_frames - 1) if transition_frames > 1 else 1.0
        frame_a = frames_a[len(frames_a) - transition_frames + i]
        frame_b = frames_b[i]
        blended = Image.blend(frame_a, frame_b, alpha)
        result.append(blended)

    result.extend(frames_b[transition_frames:])
    return result


def slide_transition(
    frame_a: Image.Image,
    frame_b: Image.Image,
    num_frames: int,
    direction: str = "left",
) -> list[Image.Image]:
    """Create a slide transition between two frames.

    Args:
        frame_a: Outgoing frame.
        frame_b: Incoming frame.
        num_frames: Number of transition frames.
        direction: 'left', 'right', 'up', 'down'.

    Returns:
        List of transition frames.
    """
    w, h = frame_a.size
    frames = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        t = _ease_in_out(t)

        canvas = Image.new("RGB", (w, h), (0, 0, 0))

        if direction == "left":
            offset = int(w * t)
            canvas.paste(frame_a, (-offset, 0))
            canvas.paste(frame_b, (w - offset, 0))
        elif direction == "right":
            offset = int(w * t)
            canvas.paste(frame_a, (offset, 0))
            canvas.paste(frame_b, (-w + offset, 0))
        elif direction == "up":
            offset = int(h * t)
            canvas.paste(frame_a, (0, -offset))
            canvas.paste(frame_b, (0, h - offset))
        elif direction == "down":
            offset = int(h * t)
            canvas.paste(frame_a, (0, offset))
            canvas.paste(frame_b, (0, -h + offset))

        frames.append(canvas)

    return frames


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out curve."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2
