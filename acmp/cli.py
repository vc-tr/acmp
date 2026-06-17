"""ACMP v2 Command-Line Interface."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from acmp import __version__
from acmp.config import PipelineConfig


@click.group()
@click.version_option(version=__version__)
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
    "--quality", "-q",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="fast",
    help="AI animation quality: fast (14 steps), balanced (20), quality (25).",
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
    quality: str,
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
            quality=quality,
        )
        click.echo(f"\nVideo saved to: {result}")
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort() from e


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
        import subprocess

        from acmp.video.assembler import _find_ffmpeg
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


@main.command(name="eval")
@click.option("--weights", default=None, type=click.Path(exists=True),
              help="Optional trained YOLO weights to include in the benchmark.")
@click.option("--n", default=20, help="Pages per condition (clean & degraded).")
@click.option("--iou", default=0.5, help="IoU threshold for a true positive.")
@click.option("--seed", default=0, help="RNG seed for the synthetic eval set.")
def eval_cmd(weights: str | None, n: int, iou: float, seed: int):
    """Benchmark panel detectors on a synthetic labeled set.

    Always evaluates the OpenCV heuristic; add --weights to also benchmark a
    trained YOLO detector. Reports precision/recall/F1/AP@0.5/mIoU on clean and
    degraded (noisy) pages.
    """
    logging.getLogger("acmp.panels.detector").setLevel(logging.WARNING)
    from acmp.eval.runner import benchmark_suite, format_results, heuristic_detector
    from acmp.eval.synthetic import generate_split

    split = generate_split(n_train=0, n_val=0, n_test=n, seed=seed)
    detectors = {"heuristic (OpenCV)": heuristic_detector}
    if weights:
        from acmp.panels.yolo_detector import YoloPanelDetector
        detectors["learned (YOLOv8)"] = YoloPanelDetector(weights)

    conditions = {"clean": split["test_clean"], "degraded": split["test_degraded"]}
    results = benchmark_suite(detectors, conditions, iou_threshold=iou)
    click.echo(format_results(results))


@main.command(name="train-detector")
@click.option("--epochs", default=35, help="Training epochs.")
@click.option("--imgsz", default=512, help="Training image size.")
@click.option("--out", default="runs/panel_detector", help="Output directory.")
@click.option("--n-train", default=72, help="Number of synthetic training pages.")
@click.option("--device", default="auto", help="auto | mps | cpu | cuda index.")
def train_detector_cmd(epochs: int, imgsz: int, out: str, n_train: int, device: str):
    """Train a YOLOv8 panel detector on synthetic data and benchmark it.

    Generates a labeled split, exports YOLO format, fine-tunes YOLOv8n, then
    compares it against the OpenCV heuristic on held-out clean & degraded pages.
    Requires the training extra: pip install -e ".[train]"
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    from acmp.eval.train import train_panel_detector
    result = train_panel_detector(
        out_dir=out, epochs=epochs, imgsz=imgsz, n_train=n_train, device=device
    )
    click.echo("\n" + result["report"])


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind host.")
@click.option("--port", default=8000, help="Bind port.")
@click.option("--reload", is_flag=True, default=False, help="Auto-reload (dev).")
def serve(host: str, port: int, reload: bool):
    """Run the ACMP REST API server (FastAPI + uvicorn).

    Then open http://HOST:PORT/docs for interactive API docs.
    Requires the api extra: pip install -e ".[api]"
    """
    try:
        import uvicorn
    except ImportError as e:
        raise click.ClickException(
            'uvicorn not installed. Install with: pip install -e ".[api]"'
        ) from e
    click.echo(f"Serving ACMP API at http://{host}:{port}  (docs: /docs)")
    uvicorn.run("acmp.api.app:app", host=host, port=port, reload=reload)


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
        import json
        import urllib.request
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
