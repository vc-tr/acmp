"""In-process job store and worker for asynchronous video rendering.

The pipeline is CPU/GPU-heavy, so jobs run on a bounded ``ThreadPoolExecutor``
(default a single worker) — requests return immediately with a job id, and the
work is serialized in the background to avoid OOM on small machines.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    """A single video-rendering job and its lifecycle state."""

    id: str
    input_dir: Path
    output_path: Path
    params: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    message: str = "queued"
    error: str | None = None

    def result_available(self) -> bool:
        return self.status is JobStatus.DONE and self.output_path.exists()

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            "params": self.params,
            "result_available": self.result_available(),
        }


class JobStore:
    """Thread-safe registry of jobs with a bounded background worker pool."""

    def __init__(self, root: str | Path, max_workers: int = 1):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def create(self, params: dict) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.root / job_id
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        job = Job(
            id=job_id,
            input_dir=input_dir,
            output_path=job_dir / "output.mp4",
            params=params,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def submit(self, job: Job, runner: Callable[[Job], None]) -> None:
        """Queue ``runner(job)`` on the worker pool and track status."""
        self._executor.submit(self._run, job, runner)

    def _run(self, job: Job, runner: Callable[[Job], None]) -> None:
        job.status = JobStatus.RUNNING
        job.message = "processing"
        try:
            runner(job)
            job.status = JobStatus.DONE
            job.message = "completed"
        except Exception as e:  # noqa: BLE001 - surface any failure as job error
            logger.exception("Job %s failed", job.id)
            job.status = JobStatus.ERROR
            job.error = str(e)
            job.message = "failed"

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
