"""ACMP Command-Line Interface."""

from __future__ import annotations

import logging
import click
from pathlib import Path

from acmp.config import PipelineConfig


@click.group()
@click.version_option(version="0.1.0")
def main():
    """ACMP - Animated Comics/Manga-Manhwa Panels.

    Turn static comic/manga/manhwa chapters into animated motion comic videos.
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
    "--depth/--no-depth",
    default=False,
    help="Enable depth estimation for parallax layers (requires torch).",
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
    help="Duration per panel in seconds (overrides config).",
)
@click.option(
    "--fps",
    type=int,
    default=None,
    help="Output frames per second (overrides config).",
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
    depth: bool,
    reading_order: str,
    seconds_per_panel: float | None,
    fps: int | None,
    verbose: bool,
):
    """Process a comic chapter into an animated video."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = Path(config) if config else None
    cfg = PipelineConfig.load(config_path)

    # Apply CLI overrides
    if reading_order != "auto":
        cfg.input.reading_order = reading_order
    if seconds_per_panel is not None:
        cfg.animation.seconds_per_panel = seconds_per_panel
    if fps is not None:
        cfg.output.fps = fps

    # Ensure output has .mp4 extension
    output_path = Path(output)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    # Run pipeline
    from acmp.pipeline import process_chapter

    try:
        result = process_chapter(
            input_path=input,
            output_path=output_path,
            config=cfg,
            use_depth=depth,
        )
        click.echo(f"\nVideo saved to: {result}")
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@main.command()
def info():
    """Show system info and available features."""
    click.echo("ACMP - Animated Comics/Manga-Manhwa Panels v0.1.0\n")

    # Check dependencies
    click.echo("Dependencies:")
    _check_dep("opencv-python", "cv2")
    _check_dep("Pillow", "PIL")
    _check_dep("moviepy", "moviepy")
    _check_dep("PyMuPDF", "fitz")
    _check_dep("click", "click")
    _check_dep("tqdm", "tqdm")
    _check_dep("numpy", "numpy")
    _check_dep("pyyaml", "yaml")

    click.echo("\nOptional (depth/AI):")
    _check_dep("torch", "torch")
    _check_dep("timm", "timm")
    _check_dep("diffusers", "diffusers")

    # Check FFmpeg
    click.echo("\nFFmpeg:")
    try:
        from acmp.video.assembler import _find_ffmpeg
        exe = _find_ffmpeg()
        import subprocess
        result = subprocess.run([exe, "-version"], capture_output=True, text=True)
        version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
        click.echo(f"  ffmpeg: {version_line}")
        click.echo(f"  path: {exe}")
    except RuntimeError:
        click.echo("  ffmpeg: NOT FOUND (install with: brew install ffmpeg)")

    # Check device
    click.echo("\nCompute device:")
    try:
        import torch
        if torch.backends.mps.is_available():
            click.echo("  Apple Silicon MPS: available")
        if torch.cuda.is_available():
            click.echo(f"  CUDA: available ({torch.cuda.get_device_name(0)})")
        if not torch.backends.mps.is_available() and not torch.cuda.is_available():
            click.echo("  CPU only")
    except ImportError:
        click.echo("  torch not installed (CPU-only mode)")


def _check_dep(name: str, import_name: str):
    """Check if a Python package is available."""
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "installed")
        click.echo(f"  {name}: {version}")
    except ImportError:
        click.echo(f"  {name}: NOT INSTALLED")


if __name__ == "__main__":
    main()
