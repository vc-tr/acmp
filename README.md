# ACMP - Animated Comics/Manga-Manhwa Panels

Turn static comic, manga, and manhwa chapters into animated motion comic videos.

## What it does

ACMP takes chapter images (or a PDF) as input and produces a 9:16 vertical MP4 video with:

- **Ken Burns effects** — slow zoom and pan across panels
- **Parallax depth** — 2.5D layer separation for foreground/background movement (requires `torch`)
- **Smooth transitions** — crossfade between panels
- **Auto panel detection** — finds and orders panels automatically
- **Multi-format support** — manga (B&W, RTL), manhwa (color, vertical scroll), western comics (LTR)

## Installation

```bash
git clone https://github.com/vc-tr/acmp.git
cd acmp
pip install -e .
```

For depth-based parallax animation (optional):

```bash
pip install -e ".[depth]"
```

## Usage

```bash
# From a directory of chapter images
acmp process --input ./chapter_pages/ --output video.mp4

# From a PDF
acmp process --input chapter.pdf --output video.mp4

# With depth-based parallax (requires torch)
acmp process --input ./chapter/ --output video.mp4 --depth

# Custom timing
acmp process --input ./chapter/ --output video.mp4 --seconds-per-panel 3.0 --fps 30

# Check system info and dependencies
acmp info
```

## Configuration

Default settings are in `configs/default.yaml`. Override with `--config`:

```bash
acmp process --input ./chapter/ --output video.mp4 --config my_config.yaml
```

Key settings:

| Setting | Default | Description |
|---|---|---|
| `animation.seconds_per_panel` | 4.0 | Duration per panel |
| `animation.transition_duration` | 0.8 | Crossfade duration between panels |
| `animation.ken_burns.zoom_range` | [1.0, 1.15] | Zoom start/end |
| `output.resolution` | [1080, 1920] | 9:16 vertical (TikTok/Reels) |
| `output.fps` | 24 | Frames per second |
| `input.reading_order` | auto | auto, rtl, ltr, or vertical |

## Pipeline

```
Input (images/PDF)
  → Page extraction
  → Panel detection (OpenCV contours)
  → [Optional] Depth estimation (MiDaS)
  → [Optional] Layer separation + inpainting
  → Animation (Ken Burns / Parallax)
  → Transitions (crossfade)
  → Video encoding (FFmpeg → MP4)
```

## Requirements

- Python 3.10+
- FFmpeg (bundled via `imageio-ffmpeg`, or system install)
- Optional: PyTorch with MPS (Apple Silicon) or CUDA for depth estimation

## Project Structure

```
acmp/
├── ingest/        # Image/PDF loading
├── panels/        # Panel detection
├── depth/         # Depth estimation (MiDaS)
├── layers/        # Layer separation + inpainting
├── animation/     # Ken Burns, parallax, transitions
├── video/         # FFmpeg video assembly
└── utils/         # Image utilities, reading order detection
```

## License

MIT
