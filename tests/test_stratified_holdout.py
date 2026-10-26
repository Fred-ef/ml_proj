"""Stratified validation splits for classification workflows"""

import json

import numpy as np
import pytest

from runner.engine import run_experiment
from src.model_selection import grid_search as grid_module
from src.model_selection import trials as trials_module
from src.model_selection.build import build_model
from src.model_selection.holdout import holdout_indices
from src.utils.metrics import accuracy


def _classification_config(**overrides):
    config = {
        "n_inputs": 2,
        "arch": [{"units": 1, "act": "sigmoid"}],
        "loss": "mse",
        "optim": {"type": "sgd", "lr": 0.1},
        "epochs": 1,
    }
    return {**config, **overrides}


def test_stratified_indices_preserve_proportions_and_partition_samples():
    labels = np.array([0] * 60 + [1] * 20).reshape(-1, 1)
    train, val = holdout_indices(80, 0.25, seed=7, stratify_labels=labels)

    assert len(train) == 60 and len(val) == 20
    assert set(train).isdisjoint(val)
    assert set(train).union(val) == set(range(80))
    assert np.bincount(labels[val, 0].astype(int)).tolist() == [15, 5]
    assert np.bincount(labels[train, 0].astype(int)).tolist() == [45, 15]
    replay = holdout_indices(80, 0.25, seed=7, stratify_labels=labels)
    assert np.array_equal(train, replay[0]) and np.array_equal(val, replay[1])
    assert not np.array_equal(val, holdout_indices(80, 0.25, seed=8,
                                                  stratify_labels=labels)[1])


def test_unstratified_indices_keep_the_existing_seeded_split():
    permutation = np.random.default_rng(7).permutation(10)
    train, val = holdout_indices(10, 0.2, seed=7)
    assert np.array_equal(train, permutation[:-2])
    assert np.array_equal(val, permutation[-2:])


def test_stratified_indices_round_quotas_and_keep_each_class_in_both_splits():
    labels = np.array(["A"] * 18 + ["B"] * 2)
    train, val = holdout_indices(20, 0.2, seed=1, stratify_labels=labels)
    assert len(val) == 4
    assert np.unique(labels[val], return_counts=True)[1].tolist() == [3, 1]
    assert np.unique(labels[train], return_counts=True)[1].tolist() == [15, 1]


@pytest.mark.parametrize("n_samples,labels,fraction,message", [
    (4, np.array([0, 0, 0, 1]), 0.5, "at least two samples per class"),
    (6, np.array([0, 0, 1, 1, 2, 2]), 0.2, "room for every class"),
    (6, np.array([0, 0, 1, 1, 2, 2]), 0.8, "room for every class"),
    (4, np.array([[0, 1], [0, 1], [1, 0], [1, 0]]), 0.5, "shape"),
    (4, np.array([0, 0, 1]), 0.5, "shape"),
])
def test_stratified_indices_reject_invalid_labels_or_infeasible_splits(
        n_samples, labels, fraction, message):
    with pytest.raises(ValueError, match=message):
        holdout_indices(n_samples, fraction, seed=0, stratify_labels=labels)


def test_grid_search_shares_one_stratified_split_across_configs_and_trials(monkeypatch):
    labels = np.array([0] * 24 + [1] * 8).reshape(-1, 1)
    X = np.random.default_rng(0).normal(size=(32, 2))
    seen = []
    original = grid_module.holdout_validate

    def spy(*args, **kwargs):
        seen.append((args[4].copy(), args[5].copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(grid_module, "holdout_validate", spy)
    configs = [_classification_config(),
               _classification_config(optim={"type": "sgd", "lr": 0.2})]
    grid_module.grid_search(configs, build_model, X, labels, metric="acc",
                            val_fraction=0.25, seed=3, n_trials=2, stratify=True)

    assert len(seen) == 4
    assert all(np.array_equal(tr, seen[0][0]) and np.array_equal(val, seen[0][1])
               for tr, val in seen)
    assert np.bincount(labels[seen[0][1], 0]).tolist() == [6, 2]


def test_train_stratify_creates_one_default_holdout_without_patience(monkeypatch):
    labels = np.array([0] * 24 + [1] * 8).reshape(-1, 1)
    X = np.random.default_rng(0).normal(size=(32, 2))
    seen = []
    original = trials_module.holdout_indices

    def spy(n_samples, fraction, seed=None, **kwargs):
        split = original(n_samples, fraction, seed, **kwargs)
        seen.append((fraction, kwargs.get("stratify_labels"), split))
        return split

    monkeypatch.setattr(trials_module, "holdout_indices", spy)
    config = _classification_config(seed=4, n_trials=2, stratify=True)
    summary, _ = trials_module.run_trials(config, X, labels, None, None,
                                          metrics={"acc": accuracy}, score_fn=accuracy,
                                          score_name="acc", stage="train")

    assert len(seen) == 1 and seen[0][0] == 0.2
    assert np.array_equal(seen[0][1], labels)
    train, val = seen[0][2]
    assert len(val) == 7
    assert np.bincount(labels[val, 0]).tolist() == [5, 2]
    assert "val_acc_mean" in summary and "split" in summary["seed_plan"]


def test_stratify_rejects_kfold_regression_and_assessment():
    X = np.zeros((8, 2))
    y = np.array([0] * 4 + [1] * 4).reshape(-1, 1)
    config = _classification_config()
    with pytest.raises(ValueError, match="only to holdout"):
        grid_module.grid_search([config], build_model, X, y, k=2,
                                metric="acc", stratify=True)
    with pytest.raises(ValueError, match="classification"):
        grid_module.grid_search([config], build_model, X, y,
                                metric="mee", stratify=True)
    with pytest.raises(ValueError, match="only to train"):
        trials_module.run_trials({**config, "stratify": True}, X, y, X, y,
                                 metrics={"acc": accuracy}, score_fn=accuracy,
                                 score_name="acc", stage="assess")


def test_runner_select_records_stratified_holdout(tmp_path):
    payload = {
        "seed": 2, "val_fraction": 0.25, "stratify": True,
        "n_core": 1, "plot_top": 0,
        "fixed": {"loss": "mse", "epochs": 1},
        "grid": {
            "arch": [[{"units": 1, "act": "sigmoid"}]],
            "optim": [{"type": "sgd", "lr": 0.1}],
        },
    }
    run_dir = run_experiment("monk1", "select", payload, results_root=tmp_path)
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert config["config"]["stratify"] is True
    assert summary["validation_mode"] == "holdout"
    assert summary["n_configs"] == 1


def test_runner_rejects_stratification_for_cup_and_assess_before_io(tmp_path):
    with pytest.raises(ValueError, match="classification"):
        run_experiment("cup", "select", {"stratify": True}, results_root=tmp_path)
    with pytest.raises(ValueError, match="only to train or select"):
        run_experiment("monk1", "assess", {"stratify": True}, results_root=tmp_path)
    assert list(tmp_path.iterdir()) == []
