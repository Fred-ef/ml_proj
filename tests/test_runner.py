"""Runner: task profiles, mode handlers, saved runs and index"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.metrics import METRICS, accuracy
from runner.registry import TASKS, get_task, _profile
from runner.engine import run_experiment, _is_index_scalar

ROOT = Path(__file__).resolve().parents[1]


# --- registry: TaskProfile uses METRICS --------------------------------------

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
    assert "train_acc_mean" in summary
    assert "test_acc_mean" not in summary
    assert summary["seed_plan"]["root"] == 0
    assert len(summary["seed_plan"]["trials"]) == 2
    assert summary["seed_plan"]["trials"][0] != summary["seed_plan"]["trials"][1]

    index_rows = [json.loads(l) for l in (tmp_path / "monk1" / "index.jsonl").read_text().splitlines()]
    assert len(index_rows) == 1
    assert index_rows[0]["run_id"] == run_dir.name


def _assert_assess_curves_are_on_the_test_set(assess_dir, metric):
    """history.csv holds training and test curves; the test one ends at the reported score"""
    summary = json.loads((assess_dir / "summary.json").read_text())
    with open(assess_dir / "history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["epoch", "loss", "test_loss", metric, f"test_{metric}"]
    rep = summary["representative_trial"]
    if "epochs_run" in summary:
        expected_epochs = summary["epochs_run"][rep]
    else:
        expected_epochs = json.loads((assess_dir / "config.json").read_text())["config"]["epochs"]
    assert len(rows) == expected_epochs
    assert float(rows[-1][f"test_{metric}"]) == pytest.approx(
        summary[f"per_trial_ts_{metric}"][rep])
    assert (assess_dir / "learning_curve.png").exists()


# --- engine: select then assess ----------------------------------------------

def test_run_experiment_select_then_assess_chain(tmp_path):
    select_payload = {
        "k": 2, "seed": 0,
        "fixed": {"loss": "mse", "epochs": 5, "batch_size": None},
        "grid": {
            "arch": [[{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                     {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
            "optim": [{"type": "sgd", "lr": 0.1, "momentum": 0.9},
                      {"type": "sgd", "lr": 0.05, "momentum": 0.9}],
            "reg": [None],
        },
    }
    select_dir = run_experiment("monk1", "select", select_payload,
                                tag="pytest-select", results_root=tmp_path)
    select_summary = json.loads((select_dir / "summary.json").read_text())

    assert select_summary["metric"] == "acc"
    assert "ranking" in select_summary and "aggregate" not in select_summary
    assert {"best_config", "best_epoch_median", "val_mean", "val_std"}.isdisjoint(select_summary)
    assert len(select_summary["ranking"]) == select_summary["n_configs"] == 2

    # Assess uses the explicitly selected row, even when it is not the first
    chosen = select_summary["ranking"][1]
    assess_payload = dict(chosen["config"])
    assess_payload["n_trials"] = 2
    assess_payload["val_mean"] = chosen["val_mean"]
    assess_payload["val_std"] = chosen["val_std"]
    assess_payload["cv_stop_target"] = chosen["cv_stop_target"]
    assess_payload["cv_stop_metric"] = chosen["cv_stop_metric"]

    assess_dir = run_experiment("monk1", "assess", assess_payload,
                                tag="pytest-assess", results_root=tmp_path)
    assess_summary = json.loads((assess_dir / "summary.json").read_text())

    assert {"acc_tr_mean", "acc_ts_mean", "acc_vl_mean"} <= set(assess_summary)
    assert assess_summary["acc_vl_mean"] == chosen["val_mean"]
    assert len(assess_summary["seed_plan"]["trials"]) == 2
    _assert_assess_curves_are_on_the_test_set(assess_dir, "acc")
    assert assess_summary["cv_stop_target"] == chosen["cv_stop_target"]
    assert assess_summary["cv_stop_metric"] == "loss"
    assert len(assess_summary["epochs_run"]) == 2
    assert all(1 <= epoch <= assess_payload["epochs"] for epoch in assess_summary["epochs_run"])
    assert assess_summary["cv_stop_reached"] == [
        error <= assess_summary["cv_stop_target"]
        for error in assess_summary["cv_stop_final_errors"]
    ]


def test_run_experiment_select_with_holdout_reports_per_config_errors(tmp_path):
    select_payload = {
        "val_fraction": 0.25, "seed": 0, "n_core": 1,
        "fixed": {"loss": "mse", "epochs": 5, "batch_size": None},
        "grid": {
            "arch": [[{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                      {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
            "optim": [{"type": "sgd", "lr": 0.1, "momentum": 0.9},
                      {"type": "sgd", "lr": 0.01, "momentum": 0.9}],
            "reg": [None],
        },
    }
    run_dir = run_experiment("monk1", "select", select_payload,
                             tag="pytest-holdout", results_root=tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())

    assert summary["validation_mode"] == "holdout"
    assert summary["n_configs"] == len(summary["ranking"]) == 2
    assert {"best_config", "best_epoch_median", "val_mean", "val_std"}.isdisjoint(summary)
    ranking = summary["ranking"]
    assert [row["val_mean"] for row in ranking] == sorted(
        (row["val_mean"] for row in ranking), reverse=True)
    assert all(row["val_std"] is None for row in ranking)
    assert all(row["epochs_run"] == [5] and row["best_epoch_median"] == 5
               for row in ranking)
    assert all(row["train_error"] == pytest.approx(1 - row["train_mean"])
               and row["val_error"] == pytest.approx(1 - row["val_mean"])
               for row in ranking)
    assert "aggregate" not in summary


def test_random_select_seed_is_saved_and_can_be_replayed(tmp_path):
    payload = {
        "seed": None, "val_fraction": 0.2, "n_core": 1,
        "fixed": {"loss": "mse", "epochs": 2},
        "grid": {
            "arch": [[{"units": 2, "act": "tanh"},
                      {"units": 1, "act": "sigmoid"}]],
            "optim": [{"type": "sgd", "lr": 0.1}],
        },
    }
    first = run_experiment("monk1", "select", payload, results_root=tmp_path)
    saved = json.loads((first / "config.json").read_text())["config"]
    summary = json.loads((first / "summary.json").read_text())
    assert saved["seed_source"] == "generated"
    assert saved["seed"] == summary["seed_plan"]["root"]
    assert first.name.endswith(f"_s{saved['seed']}")

    replay = run_experiment("monk1", "select", {**payload, "seed": saved["seed"]},
                            results_root=tmp_path)
    replay_summary = json.loads((replay / "summary.json").read_text())
    assert replay_summary == summary


_TINY_MONK_ARCH = [{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                   {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]


@pytest.mark.parametrize("validation, runs", [({"val_fraction": 0.25}, 3), ({"k": 3}, 9)])
def test_run_experiment_select_with_trials_saves_spread_and_curves(tmp_path, validation, runs):
    payload = {
        **validation, "seed": 0, "n_core": 1, "n_trials": 3, "plot_top": 2,
        "fixed": {"loss": "mse", "epochs": 6, "batch_size": None},
        "grid": {
            "arch": [_TINY_MONK_ARCH],
            "optim": [{"type": "sgd", "lr": lr, "momentum": 0.9} for lr in (0.1, 0.03, 0.01)],
            "reg": [None],
        },
    }
    run_dir = run_experiment("monk1", "select", payload,
                             tag="pytest-trials", results_root=tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())

    # summary.json keeps the spread across initializations, not the (large) curves
    assert "curves" not in summary
    assert summary["n_trials"] == 3
    assert len(summary["seed_plan"]["models"]) == runs     # one stream pair per (trial, fold)
    for row in summary["ranking"]:
        assert row["n_trials"] == 3 and len(row["per_trial_val"]) == 3
        assert row["val_init_var"] == pytest.approx(np.var(row["per_trial_val"]))
        assert row["val_init_std"] == pytest.approx(np.std(row["per_trial_val"]))
        assert len(row["epochs_run"]) == runs

    # curves.json: one entry per configuration, in ranking order, 6 epochs each
    curves = json.loads((run_dir / "curves.json").read_text())
    assert [entry["rank"] for entry in curves] == [1, 2, 3]
    for entry in curves:
        for name in ("train", "val"):
            assert len(entry[name]["mean"]) == len(entry[name]["std"]) == 6
            assert entry[name]["n_alive"] == [runs] * 6    # no early stopping: nobody stops
            assert len(entry["loss"][name]["mean"]) == len(entry["loss"][name]["std"]) == 6
            assert entry["loss"][name]["n_alive"] == [runs] * 6
        assert set(entry["loss"]) == {"train", "val"}
        # a tiny net starts near 0.5, so the first MSE is about 0.5**2
        assert entry["loss"]["train"]["mean"][0] == pytest.approx(0.25, abs=0.05)

    # Plots: the first plot_top configurations, the comparison and the copy of the best one
    assert {p.name for p in (run_dir / "curves").iterdir()} == {
        "rank01.png", "rank02.png", "top_val.png"}
    assert ((run_dir / "learning_curve.png").read_bytes()
            == (run_dir / "curves" / "rank01.png").read_bytes())

    # Two panels side by side (loss and metric): much wider than tall
    from matplotlib.image import imread
    for name in ("rank01.png", "top_val.png"):
        height, width = imread(run_dir / "curves" / name).shape[:2]
        assert width > 2 * height


def test_select_null_n_trials_and_plot_top_use_their_defaults(tmp_path):
    payload = {
        "val_fraction": 0.25, "seed": 0, "n_core": 1, "n_trials": None, "plot_top": None,
        "fixed": {"loss": "mse", "epochs": 3, "batch_size": None},
        "grid": {"arch": [_TINY_MONK_ARCH], "optim": [{"type": "sgd", "lr": 0.1}], "reg": [None]},
    }
    run_dir = run_experiment("monk1", "select", payload,
                             tag="pytest-null", results_root=tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())

    assert summary["n_trials"] == 1
    assert summary["ranking"][0]["val_init_std"] is None   # one run: no spread to report
    assert (run_dir / "curves" / "rank01.png").is_file()


def test_select_plot_top_zero_saves_curves_but_no_plots(tmp_path):
    payload = {
        "val_fraction": 0.25, "seed": 0, "n_core": 1, "n_trials": 2, "plot_top": 0,
        "fixed": {"loss": "mse", "epochs": 3, "batch_size": None},
        "grid": {"arch": [_TINY_MONK_ARCH], "optim": [{"type": "sgd", "lr": 0.1}], "reg": [None]},
    }
    run_dir = run_experiment("monk1", "select", payload,
                             tag="pytest-noplots", results_root=tmp_path)
    assert (run_dir / "curves.json").is_file()
    assert not (run_dir / "curves").exists() and not (run_dir / "learning_curve.png").exists()


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
    assert "ranking" not in row and "aggregate" not in row
    assert "best_config" not in row


# --- cup: registered with a restricted mode set (no plain `train`) -----------
# CUP follows the select/assess workflow; plain train remains unavailable

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
    """select and assess (separate processes, maybe days apart) must get the
    same 100 internal test rows"""
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
    saved = json.loads((select_dir / "config.json").read_text())["config"]
    assert saved["internal_test_split"] == {"fraction": 0.2, "seed": 0}

    assert select_summary["metric"] == "mee"
    chosen = select_summary["ranking"][0]
    assert chosen["val_mean"] >= 0.0   # MEE is a distance, never negative

    assess_payload = dict(chosen["config"])
    assess_payload["epochs"] = chosen["best_epoch_median"]
    assess_payload["n_trials"] = 2
    assess_payload["val_mean"] = chosen["val_mean"]
    assess_payload["val_std"] = chosen["val_std"]

    assess_dir = run_experiment("cup", "assess", assess_payload,
                                tag="pytest-assess", results_root=tmp_path)
    assess_summary = json.loads((assess_dir / "summary.json").read_text())

    assert {"mee_tr_mean", "mee_ts_mean", "mee_vl_mean"} <= set(assess_summary)
    assert assess_summary["mee_vl_mean"] == chosen["val_mean"]
    assert assess_summary["mee_ts_mean"] >= 0.0
    _assert_assess_curves_are_on_the_test_set(assess_dir, "mee")
