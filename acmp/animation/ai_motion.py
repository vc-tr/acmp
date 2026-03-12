"""AI-enhanced motion generation (Phase 5 - future implementation).

Placeholder module for AI image-to-video models like Wan 2.1, SVD, or AnimateDiff.
"""

from __future__ import annotations

from PIL import Image


def generate_ai_motion(
    panel: Image.Image,
    prompt: str = "subtle breathing motion, slight hair movement",
    num_frames: int = 24,
    model: str = "wan2.1",
) -> list[Image.Image]:
    """Generate AI-enhanced animation frames from a static panel.

    TODO: Implement with Wan 2.1 or AnimateDiff.

    Args:
        panel: Source panel image.
        prompt: Motion description prompt.
        num_frames: Number of frames to generate.
        model: AI model to use.

    Returns:
        List of animated frames.
    """
    raise NotImplementedError(
        "AI motion generation is not yet implemented. "
        "This will be added in Phase 5 with Wan 2.1 / AnimateDiff support. "
        "For now, use the deterministic parallax and Ken Burns animations."
    )
