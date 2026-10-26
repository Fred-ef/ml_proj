"""K-fold cross-validation of one configuration, with per-fold scores and curves"""

from __future__ import annotations

import numpy as np

from ..utils.metrics import METRICS
from ..utils.seeding import derive_seed, resolve_seed
from .early_stopping import EarlyStopping
from .scaling import original_target_metric, prepare_split


def kfold_indices(n_samples: int, k: int, seed: int | None = None):
    """Yield (train_idx, val_idx) for each of the k folds"""
    rng = np.random.default_rng(seed)

    indices = np.arange(n_samples)

    # shuffle first, the data may be ordered
    rng.shuffle(indices)

    # array_split also works when k does not divide n_samples
    folds = np.array_split(indices, k)

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        yield train_idx, val_idx


def cross_validate(build_model, config: dict, X, Y, k: int = 5, seed: int | None = None, metric: str = "loss", trial: int = 0) -> dict:
    """Run k-fold CV for one config; return aggregated metrics (mean, std)"""

    if metric == "loss":
        metric_fn, greater_is_better = None, False
        val_key, train_key = "val_loss", "loss"
    else:
        metric_fn, greater_is_better = METRICS[metric]
        val_key, train_key = f"val_{metric}", metric

    val_scores = []
    train_scores = []
    best_epochs = []
    epochs_run = []
    train_curves = []
    val_curves = []
    train_loss_curves = []
    val_loss_curves = []
    cv_stop_train_errors = []
    cv_stop_epochs = []
    cv_stop_metric = "loss" if metric == "acc" else metric

    master_seed = resolve_seed(seed)
    split_seed = derive_seed(master_seed, "select_split")
    for fold_idx, (train_idx, val_idx) in enumerate(kfold_indices(len(X), k, split_seed)):

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        X_train, Y_train, X_val, Y_val, scaler_Y = prepare_split(
            X_train, Y_train, X_val, Y_val,
            scale=bool(config.get("scale")),
            classification=metric == "acc" or str(config.get("task", "")).startswith("monk"))

        report_fn = metric_fn
        report_name = metric if metric_fn else None
        score_train_key, score_val_key = train_key, val_key
        if scaler_Y is not None:
            original_fn = metric_fn if metric_fn else METRICS["mse"][0]
            report_fn = original_target_metric(original_fn, scaler_Y)
            if metric == "loss":
                report_name = "reported_loss"
                score_train_key, score_val_key = report_name, f"val_{report_name}"

        index = trial * k + fold_idx

        # all configurations share a fold's initialization and batch order
        model = build_model({**config,
                             "seed": derive_seed(master_seed, "select_weights", index),
                             "shuffle_seed": derive_seed(master_seed, "select_shuffle", index)})

        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', None)

        patience = config.get('patience', None)
        if patience is not None:
            min_delta = config.get('min_delta') or 0.0
            es_callback = EarlyStopping(
                patience=patience, min_delta=min_delta,
                monitor="val_loss" if metric == "acc" else score_val_key,
                mode="min" if metric == "acc" else
                "max" if greater_is_better else "min")
        else:
            es_callback = None

        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            early_stopping=es_callback,
            metrics={report_name: report_fn} if report_fn else None
        )

        # same keys as the scores below, so curves and scores share their units
        train_curves.append(history[score_train_key])
        val_curves.append(history[score_val_key])
        train_loss_curves.append(history["loss"])
        val_loss_curves.append(history["val_loss"])

        # Retraining target: training error at the first validation optimum of
        # the fold, whatever epoch the ranking uses. With scaling these keys are
        # in the original target units (reported_loss included)
        stop_train_key = "loss" if metric == "acc" else score_train_key
        stop_val_key = "val_loss" if metric == "acc" else score_val_key
        stop_pick = np.argmax if metric != "acc" and greater_is_better else np.argmin
        stop_idx = int(stop_pick(history[stop_val_key]))
        cv_stop_train_errors.append(float(history[stop_train_key][stop_idx]))
        cv_stop_epochs.append(stop_idx + 1)

        # score the last epoch, or the early stopping checkpoint
        completed = len(history[score_val_key])
        if es_callback is None:
            selected_idx = completed - 1
        else:
            pick = np.argmax if es_callback.mode == "max" else np.argmin
            selected_idx = int(pick(history[es_callback.monitor]))
        val_scores.append(history[score_val_key][selected_idx])
        train_scores.append(history[score_train_key][selected_idx])
        best_epochs.append(selected_idx + 1)
        epochs_run.append(completed)

    # plain Python values, ready for JSON
    return {
        f"val_{metric}_mean": float(np.mean(val_scores)),
        f"val_{metric}_std": float(np.std(val_scores)),
        f"train_{metric}_mean": float(np.mean(train_scores)),
        f"train_{metric}_std": float(np.std(train_scores)),
        f"best_epoch_median": int(np.median(best_epochs)),
        f"best_epochs": best_epochs,
        "epochs_run": epochs_run,
        "train_scores": [float(s) for s in train_scores],
        "val_scores": [float(s) for s in val_scores],
        "train_curves": train_curves,
        "val_curves": val_curves,
        "train_loss_curves": train_loss_curves,
        "val_loss_curves": val_loss_curves,
        "cv_stop_train_errors": cv_stop_train_errors,
        "cv_stop_epochs": cv_stop_epochs,
        "cv_stop_metric": cv_stop_metric,
    }
