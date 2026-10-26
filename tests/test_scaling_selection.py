"""Validation metrics and scaling stay in the same units across workflows"""

import numpy as np
import pytest

from src.model_selection.holdout import holdout_validate
from src.model_selection.kfold import cross_validate
from src.model_selection.scaling import prepare_split
from src.model_selection.trials import run_trials
from src.nn.losses import MSE
from src.utils.metrics import accuracy, mee


X = np.array([[0.0], [1.0], [2.0]])
Y = np.array([[-100.0, -1.0], [100.0, 1.0], [0.0, 0.0]])


class TwoEpochModel:
    """Produce errors that rank in opposite orders before/after target scaling"""

    def fit(self, x_train, y_train, *, validation_data, metrics, early_stopping, **_):
        assert early_stopping.monitor == "val_mee"
        assert early_stopping.mode == "min"
        y_val = validation_data[1]
        score = metrics["mee"]
        # Standardized errors [0.2, 0] and [0, 1] correspond to original
        # errors [20, 0] and [0, 1]. Original-unit MEE selects epoch 2
        offsets = (np.array([[0.2, 0.0]]), np.array([[0.0, 1.0]]))
        val_scores = [score(y_val + offset, y_val) for offset in offsets]
        train_scores = [score(y_train + offset, y_train) for offset in offsets]
        self.observed_scores = val_scores
        for epoch in range(2):
            early_stopping.should_stop(val_scores[:epoch + 1])
        return {"loss": [1.0, 1.0], "val_loss": [1.0, 1.0],
                "mee": train_scores, "val_mee": val_scores}


def test_kfold_selects_scaled_cup_epoch_using_original_mee(monkeypatch):
    from src.model_selection import kfold

    monkeypatch.setattr(kfold, "kfold_indices",
                        lambda n, k, seed: iter([(np.array([0, 1]), np.array([2]))] * k))
    models = []

    def build_model(_):
        model = TwoEpochModel()
        models.append(model)
        return model

    result = cross_validate(build_model, {"scale": True, "patience": 3, "epochs": 2},
                            X, Y, k=2, seed=0, metric="mee")

    assert result["best_epochs"] == [2, 2]
    assert result["val_mee_mean"] == pytest.approx(1.0)
    assert models[0].observed_scores == pytest.approx([20.0, 1.0])


def test_holdout_selects_scaled_cup_epoch_using_original_mee():
    model = TwoEpochModel()
    result = holdout_validate(lambda _: model,
                              {"scale": True, "patience": 3, "epochs": 2},
                              X, Y, np.array([0, 1]), np.array([2]),
                              metric="mee")

    assert result["best_epochs"] == [2]
    assert result["val_mee_mean"] == pytest.approx(1.0)
    assert model.observed_scores == pytest.approx([20.0, 1.0])


def test_train_trials_monitors_original_mee_and_records_selected_epoch(monkeypatch):
    from src.model_selection import trials

    monkeypatch.setattr(trials, "holdout_indices",
                        lambda n, fraction, seed: (np.array([0, 1]), np.array([2])))

    class TrialModel(TwoEpochModel):
        loss = MSE()

        def predict(self, x):
            return np.zeros((len(x), 2))

    models = []

    def build_model(_):
        model = TrialModel()
        models.append(model)
        return model

    monkeypatch.setattr(trials, "build_model", build_model)
    summary, histories = run_trials(
        {"scale": True, "patience": 3, "min_delta": None,
         "epochs": 2, "n_trials": 1, "seed": 0},
        X, Y, None, None, metrics={}, score_fn=mee, score_name="mee", stage="train")

    assert models[0].observed_scores == pytest.approx([20.0, 1.0])
    assert histories[0]["val_mee"] == pytest.approx([20.0, 1.0])
    assert summary["selected_epochs"] == [2]
    assert summary["epochs_run"] == [2]


def test_scalers_are_fitted_only_on_training_data():
    x_train, y_train, x_val, y_val, _ = prepare_split(
        np.array([[0.0], [2.0]]), Y[:2],
        np.array([[10.0]]), np.array([[1000.0, 5.0]]),
        scale=True, classification=False)

    assert x_train[:, 0] == pytest.approx([-1.0, 1.0])
    assert x_val[0, 0] == pytest.approx(9.0)
    assert y_train[:, 0] == pytest.approx([-1.0, 1.0])
    assert y_val[0] == pytest.approx([10.0, 5.0])


@pytest.mark.parametrize("workflow", ["holdout", "kfold", "trials"])
def test_monk_scaling_is_rejected(workflow):
    y = np.array([[0.0], [1.0], [0.0]])
    config = {"scale": True, "epochs": 2}
    with pytest.raises(ValueError, match="scaling MONK data is unsupported"):
        if workflow == "holdout":
            holdout_validate(None, config, X, y, np.array([0, 1]),
                             np.array([2]), metric="acc")
        elif workflow == "kfold":
            cross_validate(None, config, X, y, k=2, metric="acc")
        else:
            run_trials(config, X, y, X, y, metrics={}, score_fn=accuracy,
                       score_name="acc", stage="assess")


def test_monk_task_rejects_scaling_with_generic_loss_metric():
    y = np.array([[0.0], [1.0], [0.0]])
    with pytest.raises(ValueError, match="scaling MONK data is unsupported"):
        holdout_validate(None, {"task": "monk1", "scale": True}, X, y,
                         np.array([0, 1]), np.array([2]), metric="loss")
