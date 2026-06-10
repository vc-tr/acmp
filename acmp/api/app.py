"""FastAPI application exposing the ACMP pipeline as a REST service.

Endpoints
---------
* ``GET  /health``              — liveness + version
* ``POST /jobs``                — upload pages (images or a PDF) + params -> job id
* ``GET  /jobs``                — list jobs
* ``GET  /jobs/{id}``           — job status
* ``GET  /jobs/{id}/result``    — download the rendered MP4
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from acmp import __version__
from acmp.api.jobs import Job, JobStore

logger = logging.getLogger(__name__)

# Accept the same image types the loader supports, plus PDF.
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".pdf"}


def _resolve_input_path(input_dir: Path) -> Path:
    """Return the path to hand the pipeline: a lone PDF file, else the directory."""
    files = [f for f in input_dir.iterdir() if f.is_file()]
    if len(files) == 1 and files[0].suffix.lower() == ".pdf":
        return files[0]
    return input_dir


def _make_runner():
    """Build the job runner that invokes the pipeline (imported lazily)."""

    def runner(job: Job) -> None:
        from acmp.config import PipelineConfig
        from acmp.pipeline import process_chapter

        p = job.params
        cfg = PipelineConfig.load()
        cfg.animation.seconds_per_panel = float(p.get("seconds_per_panel", 4.0))
        cfg.output.fps = int(p.get("fps", 24))
        if p.get("reading_order", "auto") != "auto":
            cfg.input.reading_order = p["reading_order"]

        process_chapter(
            input_path=_resolve_input_path(job.input_dir),
            output_path=job.output_path,
            config=cfg,
            use_ai=bool(p.get("use_ai", False)),
            llm_prefer=p.get("llm", "fallback"),
            quality=p.get("quality", "fast"),
        )

    return runner


def create_app(
    jobs_root: str | Path | None = None,
    max_workers: int = 1,
    runner=None,
) -> FastAPI:
    """Build the FastAPI app. ``runner`` is injectable for testing (defaults to
    the real pipeline runner)."""
    app = FastAPI(
        title="ACMP API",
        version=__version__,
        description="Turn static comic/manga/manhwa pages into animated motion-comic videos.",
    )
    store = JobStore(jobs_root or (Path(tempfile.gettempdir()) / "acmp_jobs"), max_workers=max_workers)
    runner = runner or _make_runner()
    app.state.store = store

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/jobs", status_code=201)
    async def create_job(
        files: list[UploadFile] = File(..., description="Page images, or a single PDF."),
        use_ai: bool = Form(False),
        llm: str = Form("fallback"),
        seconds_per_panel: float = Form(4.0),
        fps: int = Form(24),
        reading_order: str = Form("auto"),
        quality: str = Form("fast"),
    ) -> dict:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        params = {
            "use_ai": use_ai,
            "llm": llm,
            "seconds_per_panel": seconds_per_panel,
            "fps": fps,
            "reading_order": reading_order,
            "quality": quality,
        }
        job = store.create(params)

        saved = 0
        for f in files:
            name = Path(f.filename or "").name
            if not name or Path(name).suffix.lower() not in _ALLOWED_SUFFIXES:
                continue
            (job.input_dir / name).write_bytes(await f.read())
            saved += 1

        if saved == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No supported files. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
            )

        store.submit(job, runner)
        return job.to_dict()

    @app.get("/jobs")
    def list_jobs() -> dict:
        return {"jobs": [j.to_dict() for j in store.list()]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job.to_dict()

    @app.get("/jobs/{job_id}/result")
    def get_result(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if not job.result_available():
            raise HTTPException(
                status_code=409,
                detail=f"Result not ready (status: {job.status.value}).",
            )
        return FileResponse(job.output_path, media_type="video/mp4", filename=f"{job_id}.mp4")

    return app


# Module-level app for `uvicorn acmp.api.app:app`.
app = create_app()
