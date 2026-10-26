"""Final CUP training uses the full labelled data and the preset trial-0 seed."""

import json

import numpy as np
import pytest

from experiments.cup_final import DEFAULT_TRAINING_CSV, train_final_cup
from runner.engine import run_experiment
from src.data.cup import load_cup_train
from src.utils.metrics import mee
from src.utils.model_artifact import load_model_artifact


def test_final_cup_refits_on_all_rows_and_saves_loadable_artifact(tmp_path):
    assessment = run_experiment(
        "cup", "assess", {
            "seed": 71,
            "n_trials": 2,
            "n_inputs": 12,
            "arch": [{"units": 4, "act": "relu"},
                     {"units": 4, "act": "identity"}],
            "init": "he",
            "loss": "mse",
            "optim": {"type": "sgd", "lr": 0.0001},
            "epochs": 3,
            "batch_size": None,
            "scale": True,
            "cv_stop_target": 1e9,
            "cv_stop_metric": "mee",
        }, results_root=tmp_path,
    )
    assess_summary = json.loads((assessment / "summary.json").read_text())
    trial_zero = assess_summary["seed_plan"]["trials"][0]

    final_run = train_final_cup(assessment, results_root=tmp_path)
    saved = json.loads((final_run / "summary.json").read_text())
    model, config, scaler_x, scaler_y, metadata = load_model_artifact(
        final_run / "model.npz")
    _ids, X, y = load_cup_train(str(DEFAULT_TRAINING_CSV))

    assert saved["training_rows"] == metadata["training_rows"] == 500
    assert saved["selected_trial"] == 0
    assert saved["epochs_run"] == 1
    assert saved["cv_stop_reached"] is True
    assert config["seed"] == trial_zero["weights"]
    assert config["shuffle_seed"] == trial_zero["shuffle"]
    np.testing.assert_allclose(scaler_x.mean, X.mean(axis=0))
    np.testing.assert_allclose(scaler_y.mean, y.mean(axis=0))
    pred = scaler_y.inverse_transform(model.predict(scaler_x.transform(X)))
    assert mee(pred, y) == pytest.approx(saved["train_mee"])
    assert (final_run / "history.csv").exists()


def test_final_cup_rejects_non_assessment_run(tmp_path):
    run = tmp_path / "wrong-run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"config": {"task": "cup", "mode": "select"}}))
    (run / "summary.json").write_text(json.dumps({"seed_plan": {"trials": []}}))
    with pytest.raises(ValueError, match="completed CUP assess"):
        train_final_cup(run, results_root=tmp_path)
