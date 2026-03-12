"""LLM-based scene analysis for comic panels.

Analyzes each panel to understand characters, actions, emotions, and generates
motion prompts for AI animation. Uses Claude API (primary) with Ollama fallback.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from PIL import Image

from acmp.scene.prompts import (
    PANEL_ANALYSIS_SYSTEM,
    PANEL_ANALYSIS_USER,
    PANEL_CONTEXT_WITH_NEIGHBORS,
    PANEL_CONTEXT_STANDALONE,
)

logger = logging.getLogger(__name__)


@dataclass
class PanelAnalysis:
    """Structured analysis of a single comic panel."""
    description: str = ""
    characters: list[str] = field(default_factory=list)
    action: str = ""
    motion_intensity: str = "medium"
    emotion: str = ""
    camera_suggestion: str = ""
    motion_prompt: str = ""
    transition_to_next: str = "crossfade"

    @classmethod
    def from_dict(cls, data: dict) -> PanelAnalysis:
        return cls(
            description=data.get("description", ""),
            characters=data.get("characters", []),
            action=data.get("action", ""),
            motion_intensity=data.get("motion_intensity", "medium"),
            emotion=data.get("emotion", ""),
            camera_suggestion=data.get("camera_suggestion", ""),
            motion_prompt=data.get("motion_prompt", ""),
            transition_to_next=data.get("transition_to_next", "crossfade"),
        )

    @classmethod
    def fallback(cls, panel_idx: int = 0) -> PanelAnalysis:
        """Generate a generic fallback analysis when LLM is unavailable."""
        return cls(
            description="Comic panel with characters",
            characters=["character"],
            action="static pose",
            motion_intensity="low",
            emotion="neutral",
            camera_suggestion="slow zoom in",
            motion_prompt="subtle breathing motion, slight hair movement, gentle parallax depth effect",
            transition_to_next="crossfade",
        )


def _image_to_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Convert PIL Image to base64 string, resizing if needed to save tokens."""
    # Resize to limit API costs
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def _build_context(panel_idx: int, total_panels: int, prev_description: str | None) -> str:
    """Build context string for the analysis prompt."""
    if prev_description:
        return PANEL_CONTEXT_WITH_NEIGHBORS.format(
            panel_num=panel_idx + 1,
            total_panels=total_panels,
            prev_description=prev_description,
        )
    return PANEL_CONTEXT_STANDALONE


def analyze_panel_claude(
    panel: Image.Image,
    panel_idx: int = 0,
    total_panels: int = 1,
    prev_description: str | None = None,
    api_key: str | None = None,
) -> PanelAnalysis:
    """Analyze a panel using Claude API (vision).

    Args:
        panel: Panel image (PIL RGB).
        panel_idx: Index of this panel in the chapter.
        total_panels: Total number of panels.
        prev_description: Description of the previous panel for context.
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

    Returns:
        PanelAnalysis with structured scene information.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    context = _build_context(panel_idx, total_panels, prev_description)
    img_b64 = _image_to_base64(panel)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=PANEL_ANALYSIS_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": PANEL_ANALYSIS_USER.format(context=context),
                        },
                    ],
                }
            ],
        )

        response_text = message.content[0].text.strip()

        # Handle potential markdown code blocks in response
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

        data = json.loads(response_text)
        analysis = PanelAnalysis.from_dict(data)
        logger.info(f"Panel {panel_idx + 1}: {analysis.description[:60]}...")
        return analysis

    except json.JSONDecodeError as e:
        logger.warning(f"Panel {panel_idx + 1}: Failed to parse LLM JSON response: {e}")
        return PanelAnalysis.fallback(panel_idx)
    except Exception as e:
        logger.warning(f"Panel {panel_idx + 1}: Claude API error: {e}")
        return PanelAnalysis.fallback(panel_idx)


def analyze_panel_ollama(
    panel: Image.Image,
    panel_idx: int = 0,
    total_panels: int = 1,
    prev_description: str | None = None,
    model: str = "llama3.2-vision",
    host: str = "http://localhost:11434",
) -> PanelAnalysis:
    """Analyze a panel using a local Ollama vision model (fallback).

    Args:
        panel: Panel image (PIL RGB).
        panel_idx: Index of this panel.
        total_panels: Total panels in chapter.
        prev_description: Previous panel description.
        model: Ollama model name with vision capability.
        host: Ollama server URL.

    Returns:
        PanelAnalysis with structured scene information.
    """
    import urllib.request

    context = _build_context(panel_idx, total_panels, prev_description)
    img_b64 = _image_to_base64(panel, max_size=512)  # Smaller for local model

    prompt = f"{PANEL_ANALYSIS_SYSTEM}\n\n{PANEL_ANALYSIS_USER.format(context=context)}"

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        response_text = result.get("response", "").strip()

        # Try to extract JSON from response
        if "{" in response_text:
            json_start = response_text.index("{")
            json_end = response_text.rindex("}") + 1
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            analysis = PanelAnalysis.from_dict(data)
            logger.info(f"Panel {panel_idx + 1} (Ollama): {analysis.description[:60]}...")
            return analysis

    except Exception as e:
        logger.warning(f"Panel {panel_idx + 1}: Ollama error: {e}")

    return PanelAnalysis.fallback(panel_idx)


def analyze_panel(
    panel: Image.Image,
    panel_idx: int = 0,
    total_panels: int = 1,
    prev_description: str | None = None,
    prefer: str = "claude",
    api_key: str | None = None,
) -> PanelAnalysis:
    """Analyze a panel with automatic fallback.

    Tries Claude API first, falls back to Ollama, then to generic fallback.

    Args:
        panel: Panel image.
        panel_idx: Panel index.
        total_panels: Total panels.
        prev_description: Previous panel description.
        prefer: 'claude', 'ollama', or 'fallback'.
        api_key: Anthropic API key (optional).

    Returns:
        PanelAnalysis with structured scene information.
    """
    if prefer == "fallback":
        return PanelAnalysis.fallback(panel_idx)

    # Try Claude API first
    if prefer in ("claude", "auto"):
        try:
            return analyze_panel_claude(panel, panel_idx, total_panels, prev_description, api_key)
        except ImportError:
            logger.info("anthropic package not installed, trying Ollama...")
        except Exception as e:
            logger.warning(f"Claude API failed: {e}, trying Ollama...")

    # Try Ollama fallback
    if prefer in ("ollama", "claude", "auto"):
        try:
            return analyze_panel_ollama(panel, panel_idx, total_panels, prev_description)
        except Exception as e:
            logger.warning(f"Ollama failed: {e}, using generic fallback")

    return PanelAnalysis.fallback(panel_idx)


def analyze_chapter(
    panels: list[Image.Image],
    prefer: str = "claude",
    api_key: str | None = None,
) -> list[PanelAnalysis]:
    """Analyze all panels in a chapter sequentially.

    Each panel gets context from the previous panel's analysis for narrative flow.

    Args:
        panels: List of cropped panel images.
        prefer: LLM preference ('claude', 'ollama', 'fallback').
        api_key: Anthropic API key (optional).

    Returns:
        List of PanelAnalysis, one per panel.
    """
    analyses = []
    prev_desc = None

    for i, panel in enumerate(panels):
        logger.info(f"Analyzing panel {i + 1}/{len(panels)}...")
        analysis = analyze_panel(
            panel=panel,
            panel_idx=i,
            total_panels=len(panels),
            prev_description=prev_desc,
            prefer=prefer,
            api_key=api_key,
        )
        analyses.append(analysis)
        prev_desc = analysis.description

    return analyses
