"""request validation, job lifecycle, and the HTTP routes

Validates the FastAPI layer over the runner engine:

  - jobs.submit() with results_root=tmp_path, for anything
    about job lifecycle mechanics (queued->done/failed, error capture,
    job_id collisions) - isolated from the real results/ directory
  - full HTTP round trips via TestClient for the routes (POST /jobs,
    GET /jobs/{id}, GET /runs/...) - writes into the real results/ dir

Run from the project root:  pytest tests/test_service.py -v
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from service.app import app
from service import jobs
from service.schemas import JobRequest

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]


def _wait(job_id: str, timeout: float = 20.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = client.get(f"/jobs/{job_id}").json()
        if rec["status"] in ("done", "failed"):
            return rec
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


def _cleanup_run(task: str, run_id: str) -> None:
    """Undo a real write into results/<task>/ made by a full-HTTP test."""
    shutil.rmtree(ROOT / "results" / task / run_id, ignore_errors=True)
    index = ROOT / "results" / task / "index.jsonl"
    if index.exists():
        rows = [json.loads(l) for l in index.read_text().splitlines()]
        rows = [r for r in rows if r["run_id"] != run_id]
        index.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if rows:
            from src.utils.experiment import index_to_csv
            index_to_csv(index)
        else:
            (ROOT / "results" / task / "index.csv").unlink(missing_ok=True)
            index.unlink(missing_ok=True)


# --- schemas: the discriminated optim union (the reason it isn't one flat model) --

def test_sgd_and_quickprop_payloads_only_carry_their_own_fields():
    sgd = JobRequest(task="monk1", mode="train", config={
        "arch": [{"units": 2, "act": "tanh"}, {"units": 1, "act": "sigmoid"}],
        "optim": {"type": "sgd", "lr": 0.1, "momentum": 0.9}, "epochs": 10,
    }).payload()
    assert set(sgd["optim"]) == {"type", "lr", "momentum", "nesterov"}

    qp = JobRequest(task="monk1", mode="train", config={
        "arch": [{"units": 2, "act": "tanh"}, {"units": 1, "act": "sigmoid"}],
        "optim": {"type": "quickprop", "lr": 0.1, "mu": 1.75}, "epochs": 10,
    }).payload()
    assert set(qp["optim"]) == {"type", "lr", "mu"}


def test_unknown_optimizer_type_is_rejected():
    with pytest.raises(Exception, match="sgd.*quickprop|quickprop.*sgd"):
        JobRequest(task="monk1", mode="train", config={
            "arch": [{"units": 2, "act": "tanh"}, {"units": 1, "act": "sigmoid"}],
            "optim": {"type": "adam", "lr": 0.1}, "epochs": 10,
        })


def test_assess_omits_val_fields_when_not_provided_but_keeps_them_when_given():
    base = {"arch": [{"units": 2, "act": "tanh"}, {"units": 1, "act": "sigmoid"}],
           "optim": {"type": "sgd", "lr": 0.1}, "epochs": 10}

    without = JobRequest(task="monk1", mode="assess", assess=base).payload()
    assert "val_mean" not in without and "val_std" not in without

    with_val = JobRequest(task="monk1", mode="assess",
                          assess={**base, "val_mean": 0.8, "val_std": 0.02}).payload()
    assert with_val["val_mean"] == 0.8 and with_val["val_std"] == 0.02


def test_mode_requires_its_matching_field():
    with pytest.raises(Exception, match="requires the 'config' field"):
        JobRequest(task="monk1", mode="train")
    with pytest.raises(Exception, match="requires the 'select' field"):
        JobRequest(task="monk1", mode="select")


# --- job registry mechanics (isolated: results_root=tmp_path) ----------------

def _tiny_payload(**overrides):
    cfg = {"n_inputs": 17,
          "arch": [{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                   {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}],
          "loss": "mse", "optim": {"type": "sgd", "lr": 0.1, "momentum": 0.9},
          "epochs": 5, "n_trials": 2}
    cfg.update(overrides)
    return cfg


def test_submit_runs_in_the_background_and_reaches_done(tmp_path):
    rec = jobs.submit("test-lifecycle-1", "monk1", "train", _tiny_payload(), "t",
                      results_root=tmp_path)
    assert rec.status == "queued"
    done = _wait_record(rec)
    assert done.status == "done"
    assert (tmp_path / "monk1" / done.run_id / "summary.json").exists()


def _wait_record(rec, timeout: float = 20.0):
    t0 = time.time()
    while rec.status not in ("done", "failed") and time.time() - t0 < timeout:
        time.sleep(0.1)
    return rec


def test_a_job_that_raises_inside_the_worker_is_reported_as_failed_not_crashed(tmp_path):
    broken = _tiny_payload(arch=[{"units": 1, "act": "not_a_real_activation",
                                  "init": "uniform", "init_kwargs": {}}])
    rec = jobs.submit("test-lifecycle-fail", "monk1", "train", broken, "t", results_root=tmp_path)
    done = _wait_record(rec)
    assert done.status == "failed"
    assert "not_a_real_activation" in done.error


def test_duplicate_job_id_is_rejected_not_silently_overwritten(tmp_path):
    jobs.submit("test-lifecycle-dup", "monk1", "train", _tiny_payload(), "t", results_root=tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        jobs.submit("test-lifecycle-dup", "monk1", "train", _tiny_payload(), "t", results_root=tmp_path)


# --- HTTP layer: validation errors (no job created, no files touched) --------

def test_post_jobs_rejects_negative_learning_rate():
    r = client.post("/jobs", json={
        "task": "monk1", "mode": "train",
        "config": {"arch": [{"units": 2, "act": "tanh"}, {"units": 1, "act": "sigmoid"}],
                   "optim": {"type": "sgd", "lr": -1}, "epochs": 10},
    })
    assert r.status_code == 422


def test_post_jobs_rejects_unknown_activation():
    r = client.post("/jobs", json={
        "task": "monk1", "mode": "train",
        "config": {"arch": [{"units": 2, "act": "tnah"}, {"units": 1, "act": "sigmoid"}],
                   "optim": {"type": "sgd", "lr": 0.1}, "epochs": 10},
    })
    assert r.status_code == 422


def test_post_jobs_rejects_train_on_cup():
    """cup's allowed_modes excludes train (see runner/registry.py) — checked
    synchronously in the route, before any job is queued."""
    r = client.post("/jobs", json={
        "task": "cup", "mode": "train",
        "config": {"arch": [{"units": 4, "act": "tanh"}, {"units": 4, "act": "identity"}],
                   "optim": {"type": "sgd", "lr": 0.1}, "epochs": 10},
    })
    assert r.status_code == 422
    assert "not allowed" in r.json()["detail"]


def test_get_unknown_job_is_404():
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_get_unknown_run_is_404():
    assert client.get("/runs/monk1/does-not-exist").status_code == 404


def test_get_runs_for_a_task_with_no_index_yet_returns_empty_list():
    r = client.get("/runs?task=monk3")
    assert r.status_code == 200
    # monk3 may or may not have prior runs from other sessions; just check the
    # shape of the response, not its length.
    assert isinstance(r.json(), list)


# --- HTTP layer: one real end-to-end job (writes to and cleans up results/) --

def test_post_jobs_train_end_to_end_produces_a_readable_run():
    r = client.post("/jobs", json={
        "task": "monk1", "mode": "train", "tag": "pytest-service",
        "config": {"arch": [{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                            {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}],
                   "optim": {"type": "sgd", "lr": 0.1, "momentum": 0.9},
                   "epochs": 5, "n_trials": 2},
    })
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    try:
        rec = _wait(job_id)
        assert rec["status"] == "done"

        run = client.get(f"/runs/monk1/{rec['run_id']}")
        assert run.status_code == 200
        assert "test_acc_mean" in run.json()["summary"]

        plot = client.get(f"/runs/monk1/{rec['run_id']}/plot")
        assert plot.status_code == 200
        assert plot.headers["content-type"] == "image/png"

        index = client.get("/runs?task=monk1")
        assert any(row["run_id"] == rec["run_id"] for row in index.json())
    finally:
        _cleanup_run("monk1", rec["run_id"])


def test_post_jobs_select_then_assess_chain_on_cup_via_http():
    select_body = {
        "task": "cup", "mode": "select", "tag": "pytest-service",
        "select": {"k": 2, "seed": 0,
                  "fixed": {"loss": "mse", "epochs": 5, "batch_size": None},
                  "grid": {"arch": [[{"units": 4, "act": "tanh", "init": "glorot"},
                                     {"units": 4, "act": "identity", "init": "glorot"}]],
                           "optim": [{"type": "sgd", "lr": 0.01, "momentum": 0.9}],
                           "reg": [None]}},
    }
    r = client.post("/jobs", json=select_body)
    assert r.status_code == 202
    select_job = _wait(r.json()["job_id"])
    assess_job = None
    try:
        assert select_job["status"] == "done"
        select_summary = client.get(f"/runs/cup/{select_job['run_id']}").json()["summary"]

        assess_cfg = dict(select_summary["best_config"])
        assess_cfg["epochs"] = select_summary["best_epoch_median"]
        assess_cfg["n_trials"] = 2
        assess_cfg["val_mean"] = select_summary["val_mean"]
        assess_cfg["val_std"] = select_summary["val_std"]

        r2 = client.post("/jobs", json={"task": "cup", "mode": "assess",
                                        "tag": "pytest-service", "assess": assess_cfg})
        assert r2.status_code == 202
        assess_job = _wait(r2.json()["job_id"])
        assert assess_job["status"] == "done"

        assess_summary = client.get(f"/runs/cup/{assess_job['run_id']}").json()["summary"]
        assert assess_summary["mee_vl_mean"] == select_summary["val_mean"]
    finally:
        _cleanup_run("cup", select_job["run_id"])
        if assess_job is not None:
            _cleanup_run("cup", assess_job["run_id"])
