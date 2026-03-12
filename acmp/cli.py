"""ACMP v2 Command-Line Interface."""

from __future__ import annotations

import logging
import click
from pathlib import Path

from acmp.config import PipelineConfig


@click.group()
@click.version_option(version="0.2.0")
def main():
    """ACMP - Animated Comics/Manga-Manhwa Panels.

    Turn static comic/manga/manhwa chapters into animated motion comic videos.
    v2: AI-powered animation with Wan VACE + LLM scene analysis.
    """
    pass


@main.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input path: directory of images or a PDF file.",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output video file path (MP4).",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default=None,
    help="Custom YAML config file path.",
)
@click.option(
    "--ai/--no-ai",
    default=True,
    help="Enable/disable AI animation (Wan VACE). Default: enabled.",
)
@click.option(
    "--depth/--no-depth",
    default=False,
    help="Enable depth parallax (v1 fallback mode).",
)
@click.option(
    "--llm",
    type=click.Choice(["claude", "ollama", "fallback"]),
    default="claude",
    help="LLM for scene analysis. 'claude' uses API, 'ollama' uses local model.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="Anthropic API key (or set ANTHROPIC_API_KEY env var).",
)
@click.option(
    "--reading-order",
    type=click.Choice(["auto", "rtl", "ltr", "vertical"]),
    default="auto",
    help="Reading order override.",
)
@click.option(
    "--seconds-per-panel", "-s",
    type=float,
    default=None,
    help="Duration per panel in seconds.",
)
@click.option(
    "--fps",
    type=int,
    default=None,
    help="Output frames per second.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def process(
    input: str,
    output: str,
    config: str | None,
    ai: bool,
    depth: bool,
    llm: str,
    api_key: str | None,
    reading_order: str,
    seconds_per_panel: float | None,
    fps: int | None,
    verbose: bool,
):
    """Process a comic chapter into an animated video.

    \b
    Examples:
      # AI animation with Claude scene analysis (recommended)
      acmp process -i ./chapter/ -o video.mp4

      # AI animation with local LLM (offline)
      acmp process -i ./chapter/ -o video.mp4 --llm ollama

      # v1 fallback (no AI, just Ken Burns)
      acmp process -i ./chapter/ -o video.mp4 --no-ai

      # From PDF
      acmp process -i chapter.pdf -o video.mp4
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(config) if config else None
    cfg = PipelineConfig.load(config_path)

    if reading_order != "auto":
        cfg.input.reading_order = reading_order
    if seconds_per_panel is not None:
        cfg.animation.seconds_per_panel = seconds_per_panel
    if fps is not None:
        cfg.output.fps = fps

    output_path = Path(output)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    from acmp.pipeline import process_chapter

    try:
        result = process_chapter(
            input_path=input,
            output_path=output_path,
            config=cfg,
            use_ai=ai,
            use_depth=depth,
            llm_prefer=llm,
            api_key=api_key,
        )
        click.echo(f"\nVideo saved to: {result}")
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


@main.command()
def info():
    """Show system info, available features, and model status."""
    click.echo("ACMP v2 - Animated Comics/Manga-Manhwa Panels\n")

    click.echo("Core dependencies:")
    _check_dep("opencv-python", "cv2")
    _check_dep("Pillow", "PIL")
    _check_dep("moviepy", "moviepy")
    _check_dep("PyMuPDF", "fitz")
    _check_dep("numpy", "numpy")

    click.echo("\nAI animation:")
    _check_dep("torch", "torch")
    _check_dep("diffusers", "diffusers")
    _check_dep("accelerate", "accelerate")

    click.echo("\nScene analysis:")
    _check_dep("anthropic", "anthropic")
    _check_ollama()

    click.echo("\nFFmpeg:")
    try:
        from acmp.video.assembler import _find_ffmpeg
        import subprocess
        exe = _find_ffmpeg()
        result = subprocess.run([exe, "-version"], capture_output=True, text=True)
        version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
        click.echo(f"  ffmpeg: {version_line}")
    except RuntimeError:
        click.echo("  ffmpeg: NOT FOUND")

    click.echo("\nCompute device:")
    try:
        import torch
        if torch.backends.mps.is_available():
            click.echo("  Apple Silicon MPS: available")
        elif torch.cuda.is_available():
            click.echo(f"  CUDA: available ({torch.cuda.get_device_name(0)})")
        else:
            click.echo("  CPU only")
    except ImportError:
        click.echo("  torch not installed")

    # Check system memory
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        mem_gb = int(result.stdout.strip()) / (1024**3)
        click.echo(f"\nSystem memory: {mem_gb:.0f} GB unified")
        if mem_gb < 16:
            click.echo("  Note: 8GB is tight for AI animation. Use --no-ai as fallback.")
    except Exception:
        pass


@main.command()
def download():
    """Download AI model weights (Wan VACE 1.3B + MiDaS)."""
    click.echo("Downloading models...\n")
    from scripts.download_models import main as download_main
    download_main()


def _check_dep(name: str, import_name: str):
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "installed")
        click.echo(f"  {name}: {version}")
    except ImportError:
        click.echo(f"  {name}: NOT INSTALLED")


def _check_ollama():
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        import json
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            vision_models = [m for m in models if "vision" in m or "llava" in m]
            click.echo(f"  ollama: running ({len(models)} models)")
            if vision_models:
                click.echo(f"  vision models: {', '.join(vision_models)}")
            else:
                click.echo("  vision models: none (install with: ollama pull llama3.2-vision)")
    except Exception:
        click.echo("  ollama: not running")


if __name__ == "__main__":
    main()
