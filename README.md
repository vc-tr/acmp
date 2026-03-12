# ACMP - Animated Comics/Manga-Manhwa Panels

Turn static comic, manga, and manhwa chapters into animated motion comic videos with AI-powered character animation.

## What it does

ACMP takes chapter images (or a PDF) as input and produces a 9:16 vertical MP4 video with:

- **AI animation** — characters actually move, swing swords, turn heads (Wan VACE 1.3B image-to-video)
- **LLM scene analysis** — understands each panel's action, emotion, and motion to generate context-appropriate animation
- **Auto panel detection** — finds and orders panels automatically using OpenCV
- **Context-aware transitions** — cuts, crossfades, slides, fade-to-black chosen by scene context
- **Multi-format support** — manga (B&W, RTL), manhwa (color, vertical scroll), western comics (LTR)
- **Smart fallback** — gracefully degrades to Ken Burns/parallax when AI animation isn't available

## v2 Architecture

```
Input (images/PDF)
  → Panel Detection (OpenCV contours)
  → Scene Analysis (Claude API / Ollama / fallback)
  → AI Animation (Wan VACE 1.3B image-to-video)
  → Context-Aware Transitions
  → Video Assembly (FFmpeg → MP4)
```

The key difference from v1: instead of just zooming and panning on static panels (slideshow), v2 uses an AI image-to-video model to generate actual motion — characters move, capes flow, swords swing.

## Installation

```bash
git clone https://github.com/vc-tr/acmp.git
cd acmp
pip install -e .
```

### AI animation (recommended)

```bash
pip install -e ".[ai]"

# Download the Wan VACE 1.3B model (~3GB download, ~18GB cached)
acmp download
```

### Scene analysis

The pipeline uses an LLM to understand each panel and generate motion prompts.

- **Claude API** (recommended): Set `ANTHROPIC_API_KEY` env var or pass `--api-key`
- **Ollama** (local/offline): Install [Ollama](https://ollama.com), then `ollama pull llama3.2-vision`
- **Fallback**: Generic motion prompts when no LLM is available

### Depth parallax (optional, v1 fallback)

```bash
pip install -e ".[depth]"
```

## Usage

```bash
# AI animation with Claude scene analysis (recommended)
acmp process -i ./chapter_pages/ -o video.mp4

# AI animation with local LLM (offline, no API key needed)
acmp process -i ./chapter/ -o video.mp4 --llm ollama

# v1 fallback mode (no AI, Ken Burns zoom/pan only)
acmp process -i ./chapter/ -o video.mp4 --no-ai

# From a PDF
acmp process -i chapter.pdf -o video.mp4

# With depth parallax (v1 mode)
acmp process -i ./chapter/ -o video.mp4 --no-ai --depth

# Custom timing
acmp process -i ./chapter/ -o video.mp4 -s 3.0 --fps 30

# Check system info, dependencies, and compute device
acmp info
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `-i, --input` | (required) | Input path: directory of images or PDF |
| `-o, --output` | (required) | Output MP4 file path |
| `--ai / --no-ai` | `--ai` | Enable/disable AI animation (Wan VACE) |
| `--llm` | `claude` | LLM for scene analysis: `claude`, `ollama`, or `fallback` |
| `--api-key` | env var | Anthropic API key (or set `ANTHROPIC_API_KEY`) |
| `--reading-order` | `auto` | Panel reading order: `auto`, `rtl`, `ltr`, `vertical` |
| `-s, --seconds-per-panel` | `4.0` | Duration per panel in seconds |
| `--fps` | `24` | Output frames per second |
| `--depth / --no-depth` | `--no-depth` | Enable depth parallax (v1 fallback) |
| `-c, --config` | built-in | Custom YAML config file |
| `-v, --verbose` | off | Verbose logging |

## Configuration

Default settings are in `configs/default.yaml`. Override with `--config`:

```bash
acmp process -i ./chapter/ -o video.mp4 --config my_config.yaml
```

Key settings:

| Setting | Default | Description |
|---|---|---|
| `animation.seconds_per_panel` | 4.0 | Duration per panel |
| `animation.transition_duration` | 0.8 | Transition duration between panels |
| `output.resolution` | [1080, 1920] | 9:16 vertical (TikTok/Reels) |
| `output.fps` | 24 | Frames per second |
| `input.reading_order` | auto | auto, rtl, ltr, or vertical |

## How It Works

1. **Load** — reads images from a directory or extracts pages from a PDF
2. **Detect panels** — OpenCV contour analysis finds panel boundaries, auto-detects reading order (manga RTL, manhwa vertical, western LTR)
3. **Scene analysis** — an LLM (Claude or Ollama) analyzes each panel to understand the action, characters, emotion, and suggests camera motion and transition type
4. **AI animation** — Wan VACE 1.3B generates short video clips from each panel, guided by the LLM's motion prompts. Falls back to Ken Burns if AI fails or is disabled
5. **Transitions** — context-aware transitions (crossfade, cut, slide, fade-to-black) are applied based on scene analysis
6. **Encode** — frames are assembled into a final MP4 video via FFmpeg

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8GB (with CPU offloading) | 16GB+ |
| GPU | Apple Silicon (MPS) or NVIDIA (CUDA) | Apple M-series / RTX 3060+ |
| Disk | ~20GB (for model weights) | 30GB+ |
| Python | 3.10+ | 3.11+ |

On 8GB Apple Silicon, the pipeline uses CPU offloading and auto-downsizes resolution (480p → 320p → 256p) to fit in memory. Generation is slower (~1-3 min per panel) but works.

## Project Structure

```
acmp/
├── cli.py             # Command-line interface
├── config.py          # YAML config loading
├── pipeline.py        # Main orchestration pipeline
├── ingest/            # Image/PDF loading
├── panels/            # Panel detection (OpenCV)
├── scene/             # LLM scene analysis (Claude/Ollama)
├── animation/         # Animation engines
│   ├── wan_animator.py    # Wan VACE 1.3B AI animation
│   ├── ken_burns.py       # Ken Burns zoom/pan (fallback)
│   ├── parallax.py        # 2.5D depth parallax (fallback)
│   ├── engine.py          # v1 animation orchestrator
│   └── transitions.py     # Crossfade, slide, fade-to-black
├── depth/             # MiDaS depth estimation
├── layers/            # Layer separation + inpainting
├── video/             # FFmpeg video assembly
└── utils/             # Image utilities, reading order detection
```

## Dependencies

Core dependencies are installed automatically. Optional groups:

```bash
pip install -e ".[ai]"      # Wan VACE AI animation (diffusers, accelerate)
pip install -e ".[depth]"   # Depth estimation (torch, timm)
pip install -e ".[segment]" # Layer segmentation (torch, torchvision)
pip install -e ".[dev]"     # Development (pytest)
```

FFmpeg is bundled via `imageio-ffmpeg` — no system install needed.

## License

MIT
