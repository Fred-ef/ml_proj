"""In-memory job registry + single-worker process pool running the runner engine"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from runner.engine import run_experiment

_pool = ProcessPoolExecutor(max_workers=1)
_JOBS: dict[str, "JobRecord"] = {}


@dataclass
class JobRecord:
    job_id: str
    task: str
    mode: str
    status: str = "queued"          # queued -> done | failed
    run_id: Optional[str] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None

    def public(self) -> dict:
        return {
            "job_id": self.job_id, "task": self.task, "mode": self.mode,
            "status": self.status, "run_id": self.run_id, "error": self.error,
            "created": self.created, "finished": self.finished,
        }


def _target(task: str, mode: str, payload: dict, tag: str, results_root=None) -> str:
    """Executed in the worker process. Returns run_id for retrieval"""
    run_dir = run_experiment(task, mode, payload, tag=tag, results_root=results_root)
    return run_dir.name


def submit(job_id: str, task: str, mode: str, payload: dict, tag: str, results_root=None) -> JobRecord:
    if job_id in _JOBS:
        raise ValueError(f"job_id {job_id!r} already exists - retry, or use another tag")
    rec = JobRecord(job_id=job_id, task=task, mode=mode)
    _JOBS[job_id] = rec

    future: Future = _pool.submit(_target, task, mode, payload, tag, results_root)

    def _on_done(f: Future, rec: JobRecord = rec) -> None:
        rec.finished = time.time()
        try:
            rec.run_id = f.result()
            rec.status = "done"
        except Exception:
            rec.status = "failed"
            rec.error = traceback.format_exc(limit=6)

    future.add_done_callback(_on_done)
    return rec


def get(job_id: str) -> Optional[JobRecord]:
    return _JOBS.get(job_id)


def all_jobs() -> list[dict]:
    return [r.public() for r in sorted(_JOBS.values(), key=lambda r: r.created, reverse=True)]
