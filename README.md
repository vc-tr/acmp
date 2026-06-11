# ACMP — Animated Comics / Manga / Manhwa Panels

[![CI](https://github.com/vc-tr/acmp/actions/workflows/ci.yml/badge.svg)](https://github.com/vc-tr/acmp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Turn static comic, manga, and manhwa chapters into **animated motion-comic videos** (9:16 MP4 for TikTok/Reels) — with automatic panel detection, LLM scene understanding, and AI/Ken-Burns animation.

It ships as a **CLI**, a **REST API**, and a **web demo**, plus a trained-and-benchmarked panel detector — an end-to-end applied-ML system spanning **CV, NLP/LLMs, generative DL, and MLOps**.

---

## Skills showcase

This project is deliberately broad — each stage exercises a different competency expected of an AI engineer:

| Area | In this repo |
|---|---|
| **Computer Vision** | OpenCV panel detection (contours, NMS), MiDaS depth, `rembg` segmentation, a **trained YOLOv8 detector** benchmarked vs. the heuristic |
| **NLP / LLMs** | Vision-LLM scene analysis (Claude / Ollama) → structured motion prompts, with robust JSON parsing and graceful fallback |
| **Generative DL** | Wan VACE 1.3B image-to-video diffusion; character/background composite animation |
| **Evaluation** | IoU / precision / recall / F1 / AP@0.5 detection metrics, synthetic labeled data, clean-vs-degraded robustness benchmark |
| **MLOps / Serving** | FastAPI async job API, Streamlit demo, Docker, GitHub Actions CI, ruff, pytest (90+ tests) |

## Architecture

```
                      ┌──────────────────────────────────────────────┐
 Input (images/PDF) → │  Ingest → Panel Detection → Scene Analysis →  │ → MP4
                      │          (OpenCV | YOLO)     (LLM vision)      │
                      │   → Animation (Wan VACE AI | Ken Burns) →      │
                      │   → Context-aware Transitions → FFmpeg encode  │
                      └──────────────────────────────────────────────┘
        Entry points:   CLI  ·  REST API (FastAPI)  ·  Web demo (Streamlit)
```

## Install

```bash
git clone https://github.com/vc-tr/acmp.git
cd acmp
pip install -e .                 # core (CPU, Ken-Burns animation, CLI)
pip install -e ".[api]"          # + FastAPI REST server
pip install -e ".[demo]"         # + Streamlit web demo
pip install -e ".[train]"        # + ultralytics (train/benchmark the YOLO detector)
pip install -e ".[ai,depth]"     # + Wan VACE AI animation & MiDaS depth (heavy)
pip install -e ".[dev]"          # + pytest, ruff
```

> **NumPy note:** the stack is pinned to `numpy<2` (Streamlit/MoviePy/Matplotlib require it). FFmpeg is bundled via `imageio-ffmpeg`; OpenCV is the headless build (no system OpenGL needed).

## Quickstart

### CLI

```bash
# Render a chapter (fast, no AI — Ken Burns zoom/pan)
acmp process -i ./chapter_pages/ -o video.mp4 --no-ai --llm fallback

# AI animation + Claude scene analysis (needs the [ai] extra + ANTHROPIC_API_KEY)
acmp process -i ./chapter/ -o video.mp4 --llm claude

# From a PDF, custom timing
acmp process -i chapter.pdf -o video.mp4 -s 3.0 --fps 30

acmp info        # system / dependency / device report
```

### REST API

```bash
acmp serve                       # http://127.0.0.1:8000  (docs at /docs)

# Submit a job, poll, download
curl -F "files=@p1.png" -F "files=@p2.png" -F "use_ai=false" http://localhost:8000/jobs
curl http://localhost:8000/jobs/<job_id>
curl -o out.mp4 http://localhost:8000/jobs/<job_id>/result
```

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + version |
| `POST` | `/jobs` | upload pages/PDF + params → job id (async) |
| `GET` | `/jobs/{id}` | job status |
| `GET` | `/jobs/{id}/result` | download the MP4 |

### Web demo

```bash
streamlit run streamlit_app.py   # upload → preview detected panels → render video
```

Deployable to Hugging Face Spaces / Streamlit Community Cloud — see [`deploy/huggingface/DEPLOY.md`](deploy/huggingface/DEPLOY.md) for ready-made Space files and a step-by-step guide.

### Docker

```bash
docker build -t acmp .
docker run -p 8000:8000 acmp                         # REST API
docker run -p 8501:8501 acmp \
  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

## Panel detection: heuristic vs. learned

Panels can be found with a fast **OpenCV heuristic** (`panels.method: contour`, the default) or a **trained YOLOv8 detector** (`panels.method: yolo`, `panels.weights: <best.pt>`). To reproduce the model and benchmark:

```bash
pip install -e ".[train]"
acmp train-detector --device cpu          # generate data → train YOLOv8n → benchmark
acmp eval --weights runs/panel_detector/train/panel_detector/weights/best.pt
```

Both detectors are evaluated with the same IoU-matching metrics on held-out **clean** and **degraded** (noisy/blurred/compressed) synthetic pages:

| Condition | Detector | Precision | Recall | F1 | AP@0.5 | mIoU |
|---|---|---|---|---|---|---|
| clean | heuristic (OpenCV) | 1.000 | 1.000 | 1.000 | 1.000 | 0.906 |
| clean | learned (YOLOv8n) | 1.000 | 1.000 | 1.000 | 1.000 | **0.987** |
| degraded | heuristic (OpenCV) | 0.828 | 0.658 | 0.733 | 0.544 | 0.792 |
| degraded | learned (YOLOv8n) | **1.000** | **1.000** | **1.000** | **1.000** | **0.987** |

<sub>YOLOv8n, 50 epochs (early-stopped at 42), 64 synthetic train pages, 16 held-out per condition. Full report: [`benchmarks/panel_detection.md`](benchmarks/panel_detection.md).</sub>

**Takeaway:** the contour heuristic is excellent on clean scans but brittle under noise; the learned detector degrades gracefully — the classic motivation for a learned model.

> Train on **CPU** (`--device cpu`): YOLO/MPS training is unstable on Apple Silicon in this stack (silent non-convergence). MPS is fine for inference.

## Configuration

Defaults live in `configs/default.yaml`; override with `--config my.yaml`.

| Setting | Default | Description |
|---|---|---|
| `panels.method` | `contour` | `contour` (OpenCV) or `yolo` (learned) |
| `panels.weights` | `null` | trained YOLO weights when `method: yolo` |
| `animation.seconds_per_panel` | `4.0` | duration per panel |
| `output.resolution` | `[1080, 1920]` | 9:16 vertical |
| `output.fps` | `24` | frames per second |
| `input.reading_order` | `auto` | `auto`/`rtl`/`ltr`/`vertical` |

## Project structure

```
acmp/
├── cli.py            # CLI: process · eval · train-detector · serve · info
├── pipeline.py       # orchestration
├── ingest/           # image / PDF loading
├── panels/           # detection: detector.py (OpenCV) + yolo_detector.py (learned)
├── scene/            # LLM scene analysis (Claude / Ollama)
├── animation/        # Wan VACE AI, Ken Burns, parallax, transitions
├── depth/ · layers/  # MiDaS depth, rembg segmentation
├── video/            # FFmpeg assembly
├── eval/             # synthetic data, metrics, benchmark runner, YOLO training
├── api/              # FastAPI serving layer
└── demo/             # Streamlit web app
tests/                # pytest suite (90+ tests)
```

## Development

```bash
pip install -e ".[dev]"
pytest -q                    # run tests (add -m "not slow" to skip pipeline/ffmpeg tests)
pytest --cov=acmp            # with coverage
ruff check .                 # lint
```

CI runs lint + tests across Python 3.10–3.12 on every push/PR.

## Hardware

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB+ |
| GPU | Apple MPS / NVIDIA CUDA (AI animation) | M-series / RTX 3060+ |
| Disk | ~20 GB (AI model weights) | 30 GB+ |

The core pipeline (no-AI) and the API/demo run comfortably on CPU. AI animation auto-downsizes resolution and uses CPU offloading on 8 GB machines.

## License

MIT
