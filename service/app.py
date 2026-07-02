"""FastAPI service: HTTP glue over runner.engine

Run:
    uvicorn service.app:app --host 127.0.0.1 --port 8000 --reload

Then open http://127.0.0.1:8000/docs for the interactive form
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from runner.registry import get_task
from src.utils.experiment import load_index, make_run_id
from . import jobs
from .schemas import JobRequest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

app = FastAPI(title="ml_proj experiment service", version="0.1")


# --------------------------------------------------------------- jobs (write)
@app.post("/jobs", status_code=202)
def create_job(req: JobRequest) -> dict:
    profile = get_task(req.task)
    if req.mode not in profile.allowed_modes:
        raise HTTPException(
            422,
            f"mode={req.mode!r} is not allowed for task={req.task!r} "
            f"(allowed: {list(profile.allowed_modes)})",
        )

    payload = req.payload()
    job_id = make_run_id(req.tag, payload.get("seed"))
    try:
        rec = jobs.submit(job_id, req.task, req.mode, payload, req.tag)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return rec.public()


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return jobs.all_jobs()


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    rec = jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, f"job {job_id!r} unknown")
    return rec.public()


# --------------------------------------------------------- runs (read results/)
@app.get("/runs")
def list_runs(task: str) -> Response:
    index = RESULTS / task / "index.jsonl"
    if not index.exists():
        return Response(content="[]", media_type="application/json")
    df = load_index(index)
    return Response(content=df.to_json(orient="records"), media_type="application/json")


@app.get("/runs/{task}/{run_id}")
def get_run(task: str, run_id: str) -> dict:
    run_dir = RESULTS / task / run_id
    if not run_dir.exists():
        raise HTTPException(404, "run not found")
    return {
        "config": json.loads((run_dir / "config.json").read_text()),
        "summary": json.loads((run_dir / "summary.json").read_text()),
    }


@app.get("/runs/{task}/{run_id}/plot")
def get_plot(task: str, run_id: str) -> FileResponse:
    png = RESULTS / task / run_id / "learning_curve.png"
    if not png.exists():
        raise HTTPException(404, "no plot for this run (select mode has no single curve)")
    return FileResponse(png)
