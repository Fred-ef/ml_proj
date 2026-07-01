"""Runner tests: task/mode dispatch (registry) and the shared execution engine.

Pure infrastructure (Tier 3, see WORKING_AGREEMENT): these pin the CLI/API-
independent core that both frontends call — task profile resolution, mode
handlers wiring the from-scratch core (run_trials/grid_search/build_model),
and run persistence/indexing. No numerical logic is exercised beyond what
src/ already guarantees; MONK/CUP data files are used since they're already
checked into data/monk/ and data/cup/. Every run writes under pytest's
tmp_path, never into the real results/ directory.

Run from the project root:  pytest tests/test_runner.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.metrics import METRICS, accuracy
from runner.registry import TASKS, get_task, _profile
from runner.engine import run_experiment, _is_index_scalar

ROOT = Path(__file__).resolve().parents[1]


# --- registry: TaskProfile derives from the single METRICS source of truth ---

def test_all_registered_tasks_derive_from_the_metrics_registry():
    for profile in TASKS.values():
        fn, greater_is_better = METRICS[profile.primary]
        assert profile.metrics == {profile.primary: fn}
        assert profile.goal == ("max" if greater_is_better else "min")


def test_profile_helper_derives_consistently_from_a_metric_name():
    p = _profile("toy", 5, lambda root: None, "acc")
    assert p.metrics == {"acc": accuracy}
    assert p.primary == "acc"
    assert p.goal == "max"


def test_get_task_raises_a_clear_error_for_an_unknown_task():
    with pytest.raises(ValueError, match="monk9"):
        get_task("monk9")


# --- engine: unknown task/mode fail fast, before touching the filesystem -----

def test_run_experiment_unknown_mode_raises_before_any_io(tmp_path):
    with pytest.raises(ValueError, match="foo"):
        run_experiment("monk1", "foo", {}, results_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_run_experiment_unknown_task_raises(tmp_path):
    with pytest.raises(ValueError, match="monk9"):
        run_experiment("monk9", "train", {}, results_root=tmp_path)


# --- engine: train mode produces the expected artifacts ----------------------

def _tiny_monk1_payload(**overrides):
    cfg = {
        "arch": [{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}],
        "loss": "mse", "optim": {"type": "sgd", "lr": 0.1, "momentum": 0.9}, "reg": None,
        "epochs": 5, "batch_size": None, "seed": 0, "n_trials": 2,
    }
    cfg.update(overrides)
    return cfg


def test_run_experiment_train_produces_the_expected_artifacts(tmp_path):
    run_dir = run_experiment("monk1", "train", _tiny_monk1_payload(),
                             tag="pytest", results_root=tmp_path)

    assert run_dir.parent.name == "monk1"
    for fname in ("config.json", "summary.json", "history.csv", "learning_curve.png"):
        assert (run_dir / fname).exists()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert "test_acc_mean" in summary

    index_rows = [json.loads(l) for l in (tmp_path / "monk1" / "index.jsonl").read_text().splitlines()]
    assert len(index_rows) == 1
    assert index_rows[0]["run_id"] == run_dir.name


# --- engine: select -> assess chain, exactly as a user would run it ----------

def test_run_experiment_select_then_assess_chain(tmp_path):
    select_payload = {
        "k": 2, "seed": 0,
        "fixed": {"loss": "mse", "epochs": 5, "batch_size": None},
        "grid": {
            "arch": [[{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                     {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
            "optim": [{"type": "sgd", "lr": 0.1, "momentum": 0.9}],
            "reg": [None],
        },
    }
    select_dir = run_experiment("monk1", "select", select_payload,
                                tag="pytest-select", results_root=tmp_path)
    select_summary = json.loads((select_dir / "summary.json").read_text())

    assert select_summary["metric"] == "acc"
    assert {"val_mean", "val_std", "best_epoch_median", "best_config", "ranking"} <= set(select_summary)
    assert len(select_summary["ranking"]) == select_summary["n_configs"] == 1

    # exactly the handoff documented in F4-EXTRA: copy val_mean/val_std and
    # best_epoch_median straight from the select summary into the assess payload
    assess_payload = dict(select_summary["best_config"])
    assess_payload["epochs"] = select_summary["best_epoch_median"]
    assess_payload["n_trials"] = 2
    assess_payload["val_mean"] = select_summary["val_mean"]
    assess_payload["val_std"] = select_summary["val_std"]

    assess_dir = run_experiment("monk1", "assess", assess_payload,
                                tag="pytest-assess", results_root=tmp_path)
    assess_summary = json.loads((assess_dir / "summary.json").read_text())

    assert {"acc_tr_mean", "acc_ts_mean", "acc_vl_mean"} <= set(assess_summary)
    assert assess_summary["acc_vl_mean"] == select_summary["val_mean"]


# --- engine: the index stays a flat comparison table --------------------------

def test_is_index_scalar_keeps_flat_values_and_drops_nested_structures():
    assert _is_index_scalar(1.0) is True
    assert _is_index_scalar("x") is True
    assert _is_index_scalar([1, 2, 3]) is True
    assert _is_index_scalar(None) is True
    assert _is_index_scalar({"a": 1}) is False
    assert _is_index_scalar([{"a": 1}]) is False


def test_run_experiment_select_does_not_pollute_the_index_with_nested_fields(tmp_path):
    select_payload = {
        "k": 2, "seed": 0,
        "fixed": {"loss": "mse", "epochs": 5, "batch_size": None},
        "grid": {
            "arch": [[{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                     {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
            "optim": [{"type": "sgd", "lr": 0.1, "momentum": 0.9}],
            "reg": [None],
        },
    }
    run_experiment("monk1", "select", select_payload, tag="pytest-select", results_root=tmp_path)
    row = json.loads((tmp_path / "monk1" / "index.jsonl").read_text().splitlines()[0])
    assert "ranking" not in row and "best_config" not in row


# --- cup: registered with a restricted mode set (no plain `train`) -----------
# _load_cup's "test" slot is the internal test set: `train` would feed it to
# fit() as validation_data every epoch of every trial, which is exactly the
# internal-test peeking select/assess exist to prevent (see registry.py).

def test_cup_profile_uses_mee_and_forbids_plain_train():
    profile = TASKS["cup"]
    assert profile.n_inputs == 12
    assert profile.primary == "mee"
    assert profile.goal == "min"
    assert profile.allowed_modes == ("select", "assess")


def test_run_experiment_train_is_rejected_for_cup(tmp_path):
    with pytest.raises(ValueError, match="not allowed for task='cup'"):
        run_experiment("cup", "train", {}, results_root=tmp_path)
    assert list(tmp_path.iterdir()) == []   # rejected before any I/O


def test_cup_internal_test_split_is_deterministic_across_loads():
    """The internal test must be the SAME rows on every call: select (today)
    and assess (possibly days later, a separate process) must agree on which
    100 rows were never used for model selection."""
    _, _, X_test_1, Y_test_1 = TASKS["cup"].load(ROOT)
    _, _, X_test_2, Y_test_2 = TASKS["cup"].load(ROOT)
    assert np.array_equal(X_test_1, X_test_2)
    assert np.array_equal(Y_test_1, Y_test_2)


def _tiny_cup_select_payload(**overrides):
    payload = {
        "k": 2, "seed": 0,
        "fixed": {"loss": "mse", "epochs": 5, "batch_size": None},
        "grid": {
            "arch": [[{"units": 4, "act": "tanh", "init": "glorot"},
                     {"units": 4, "act": "identity", "init": "glorot"}]],
            "optim": [{"type": "sgd", "lr": 0.01, "momentum": 0.9}],
            "reg": [None],
        },
    }
    payload.update(overrides)
    return payload


def test_run_experiment_select_then_assess_chain_on_real_cup_data(tmp_path):
    select_dir = run_experiment("cup", "select", _tiny_cup_select_payload(),
                                tag="pytest-select", results_root=tmp_path)
    select_summary = json.loads((select_dir / "summary.json").read_text())

    assert select_summary["metric"] == "mee"
    assert select_summary["val_mean"] >= 0.0   # MEE is a distance, never negative

    assess_payload = dict(select_summary["best_config"])
    assess_payload["epochs"] = select_summary["best_epoch_median"]
    assess_payload["n_trials"] = 2
    assess_payload["val_mean"] = select_summary["val_mean"]
    assess_payload["val_std"] = select_summary["val_std"]

    assess_dir = run_experiment("cup", "assess", assess_payload,
                                tag="pytest-assess", results_root=tmp_path)
    assess_summary = json.loads((assess_dir / "summary.json").read_text())

    assert {"mee_tr_mean", "mee_ts_mean", "mee_vl_mean"} <= set(assess_summary)
    assert assess_summary["mee_vl_mean"] == select_summary["val_mean"]
    assert assess_summary["mee_ts_mean"] >= 0.0
