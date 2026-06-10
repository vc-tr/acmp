"""Tests for the FastAPI serving layer (pipeline mocked via injected runner)."""

import io
import threading
import time

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart, needed for form/file uploads
from fastapi.testclient import TestClient  # noqa: E402

from acmp.api.app import create_app  # noqa: E402


def _png_bytes(color=(120, 140, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def _client(tmp_path, runner=None) -> TestClient:
    return TestClient(create_app(jobs_root=tmp_path / "jobs", max_workers=1, runner=runner))


def _poll(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.05)
    return status


def test_health(tmp_path):
    r = _client(tmp_path).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_create_job_runs_and_returns_result(tmp_path):
    def fake_runner(job):
        job.output_path.write_bytes(b"FAKEMP4DATA")

    client = _client(tmp_path, runner=fake_runner)
    r = client.post(
        "/jobs",
        files=[("files", ("p1.png", _png_bytes(), "image/png"))],
        data={"use_ai": "false", "llm": "fallback"},
    )
    assert r.status_code == 201
    job_id = r.json()["job_id"]

    status = _poll(client, job_id)
    assert status["status"] == "done", status
    assert status["result_available"] is True

    res = client.get(f"/jobs/{job_id}/result")
    assert res.status_code == 200
    assert res.content == b"FAKEMP4DATA"


def test_unknown_job_returns_404(tmp_path):
    client = _client(tmp_path)
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/result").status_code == 404


def test_no_supported_files_returns_400(tmp_path):
    client = _client(tmp_path, runner=lambda job: None)
    r = client.post("/jobs", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400


def test_result_not_ready_returns_409(tmp_path):
    gate = threading.Event()

    def slow_runner(job):
        gate.wait(3)
        job.output_path.write_bytes(b"x")

    client = _client(tmp_path, runner=slow_runner)
    r = client.post("/jobs", files=[("files", ("p.png", _png_bytes(), "image/png"))])
    job_id = r.json()["job_id"]
    try:
        assert client.get(f"/jobs/{job_id}/result").status_code == 409  # still running
    finally:
        gate.set()


def test_list_jobs(tmp_path):
    client = _client(tmp_path, runner=lambda job: job.output_path.write_bytes(b"x"))
    client.post("/jobs", files=[("files", ("p.png", _png_bytes(), "image/png"))])
    r = client.get("/jobs")
    assert r.status_code == 200 and len(r.json()["jobs"]) >= 1
