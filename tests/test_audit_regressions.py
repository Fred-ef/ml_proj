"""Regression tests for invariants found during the project-wide audit."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from experiments.cup_final import _load_assessment_config, write_submission
from service.schemas import JobRequest
from src.data.monk import one_hot
from src.model_selection.build import build_model
from src.model_selection.early_stopping import EarlyStopping
from src.model_selection.kfold import kfold_indices
from src.model_selection.trials import run_trials
from src.nn.losses import MSE
from src.utils.metrics import mee
from src.utils.parallel import parallel_map

ROOT = Path(__file__).resolve().parents[1]


def _delayed_identity(delay: float, value: str) -> str:
    time.sleep(delay)
    return value


def _regression_config(**overrides):
    config = {
        "n_inputs": 3,
        "arch": [
            {"units": 5, "act": "tanh", "init": "glorot"},
            {"units": 2, "act": "identity", "init": "glorot"},
        ],
        "loss": "mse",
        "optim": {"type": "sgd", "lr": 0.01, "momentum": 0.0},
        "epochs": 3,
        "n_trials": 1,
    }
    config.update(overrides)
    return config


def test_loss_and_fit_reject_output_broadcasting():
    with pytest.raises(ValueError, match="same shape"):
        MSE().value(np.zeros((4, 4)), np.zeros((4, 1)))

    model = build_model(_regression_config())
    with pytest.raises(ValueError, match="target width"):
        model.fit(np.zeros((4, 3)), np.zeros((4, 1)), epochs=1)


def test_kfold_rejects_degenerate_fold_counts():
    with pytest.raises(ValueError, match="at least 2"):
        list(kfold_indices(10, 1, seed=0))
    with pytest.raises(ValueError, match="cannot exceed"):
        list(kfold_indices(3, 4, seed=0))


def test_quickprop_rejects_minibatches():
    model = build_model(_regression_config(optim={"type": "quickprop", "lr": 0.1}))
    with pytest.raises(ValueError, match="full-batch"):
        model.fit(np.zeros((4, 3)), np.zeros((4, 2)), epochs=1, batch_size=2)


def test_cross_entropy_backpropagation_matches_finite_differences():
    model = build_model({
        "n_inputs": 2,
        "arch": [{"units": 1, "act": "sigmoid", "init": "uniform",
                  "init_kwargs": {"scale": 0.2}}],
        "loss": "cross_entropy",
        "optim": {"type": "sgd", "lr": 0.1},
    })
    X = np.array([[0.2, -0.1], [0.8, 0.4], [-0.3, 0.7]])
    Y = np.array([[0.0], [1.0], [1.0]])
    assert model.gradient_check(X, Y) < 1e-5


def test_early_stopping_supports_max_metrics():
    stopper = EarlyStopping(patience=2, mode="max", monitor="val_acc")
    assert stopper.should_stop([0.5]) is False
    assert stopper.should_stop([0.5, 0.7]) is False
    assert stopper.should_stop([0.5, 0.7, 0.65]) is False
    assert stopper.should_stop([0.5, 0.7, 0.65, 0.64]) is True
    assert stopper.best_epoch == 1


def test_parallel_map_returns_input_order_not_completion_order():
    values = parallel_map(
        _delayed_identity,
        [(0.05, "first"), (0.0, "second")],
        n_core=2,
    )
    assert values == ["first", "second"]


def test_scaled_trial_history_and_summary_use_original_target_units():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    Y = 100.0 * rng.normal(size=(12, 2)) + 500.0
    summary, histories = run_trials(
        _regression_config(scale=True), X[:8], Y[:8], X[8:], Y[8:],
        metrics={"mee": mee}, score_fn=mee, score_name="mee",
    )
    assert histories[0]["mee"][-1] == pytest.approx(summary["train_mee_mean"])
    assert histories[0]["mee"][-1] > 1.0


def test_service_preserves_scale_and_rejects_assessment_early_stopping():
    base = {
        "arch": [{"units": 1, "act": "sigmoid"}],
        "optim": {"type": "sgd", "lr": 0.1},
        "epochs": 10,
        "scale": True,
    }
    payload = JobRequest(task="monk1", mode="train", config=base).payload()
    assert payload["scale"] is True

    with pytest.raises(Exception, match="patience"):
        JobRequest(task="monk1", mode="assess", assess={**base, "patience": 2})


def test_service_rejects_wrong_task_output_width_and_quickprop_minibatch():
    with pytest.raises(Exception, match="requires 4 output"):
        JobRequest(task="cup", mode="assess", assess={
            "arch": [{"units": 1, "act": "identity"}],
            "optim": {"type": "sgd", "lr": 0.1},
            "epochs": 10,
        })
    with pytest.raises(Exception, match="full-batch"):
        JobRequest(task="monk1", mode="train", config={
            "arch": [{"units": 1, "act": "sigmoid"}],
            "optim": {"type": "quickprop", "lr": 0.1},
            "epochs": 10,
            "batch_size": 2,
        })


def test_monk_one_hot_rejects_out_of_range_categories():
    invalid = np.ones((2, 6), dtype=int)
    invalid[0, 0] = 0
    with pytest.raises(ValueError, match="attribute 1"):
        one_hot(invalid)


def test_all_assessment_configs_are_fixed_epoch_only():
    paths = list((ROOT / "configs").rglob("*assess*.json"))
    assert paths
    for path in paths:
        raw = json.loads(path.read_text())
        config = raw["config"]
        assert config["epochs"] > 0
        assert not ({"patience", "min_delta", "monitor"} & config.keys()), path


def test_cup_final_config_and_submission_format(tmp_path):
    config_path = tmp_path / "cup.json"
    config_path.write_text(json.dumps({
        "task": "cup",
        "mode": "assess",
        "config": {
            "arch": [{"units": 4, "act": "identity", "init": "glorot"}],
            "optim": {"type": "sgd", "lr": 0.01},
            "epochs": 7,
        },
    }))
    assert _load_assessment_config(config_path)["epochs"] == 7

    output = write_submission(
        tmp_path / "submission.csv", "A, B", "TEAM", "17/08/2026",
        np.array([1, 2]), np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
    )
    lines = output.read_text().splitlines()
    assert lines[:4] == ["# A, B", "# TEAM", "# ML-CUP25 v1", "# 17/08/2026"]
    assert lines[4] == "1,1,2,3,4"
    assert len(lines) == 6
