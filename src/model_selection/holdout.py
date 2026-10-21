"""Evaluate one configuration on a single, shared training/validation split."""

from __future__ import annotations

import numpy as np

from ..data.preprocessing import StandardScaler
from ..utils.metrics import METRICS
from ..utils.seeding import derive_seed, resolve_seed
from .early_stopping import EarlyStopping


def holdout_indices(n_samples: int, val_fraction: float, seed: int | None = None):
    """Return shuffled train/validation indices, leaving both sets nonempty."""
    if n_samples < 2:
        raise ValueError("holdout requires at least two samples")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    indices = np.random.default_rng(seed).permutation(n_samples)
    n_val = min(max(1, int(np.ceil(n_samples * val_fraction))), n_samples - 1)
    return indices[:-n_val], indices[-n_val:]


def holdout_validate(build_model, config: dict, X, Y, train_idx, val_idx,
                     seed: int | None = None, metric: str = "loss") -> dict:
    """Train once; use the final epoch unless early stopping is enabled."""
    if metric == "loss":
        metric_fn, greater_is_better = None, False
        train_key, val_key = "loss", "val_loss"
    else:
        metric_fn, greater_is_better = METRICS[metric]
        train_key, val_key = metric, f"val_{metric}"

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    scaler_Y = None
    if config.get("scale", False):
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)

        scaler_Y = StandardScaler()
        Y_train = scaler_Y.fit_transform(Y_train)
        Y_val = scaler_Y.transform(Y_val)

    report_fn = metric_fn
    report_name = metric if metric_fn else None
    if scaler_Y is not None:
        # Model.fit operates on scaled targets. Report model-selection errors in
        # the original units so configurations with and without scaling remain
        # comparable. The optimizer still uses its ordinary training loss.
        original_fn = metric_fn if metric_fn else METRICS["mse"][0]

        def original_scale_score(y_pred, y_true):
            return original_fn(scaler_Y.inverse_transform(y_pred),
                               scaler_Y.inverse_transform(y_true))

        report_fn = original_scale_score
        if metric == "loss":
            report_name = "reported_loss"
            train_key, val_key = report_name, f"val_{report_name}"
        else:
            report_name = metric

    master_seed = resolve_seed(seed)
    model = build_model({**config,
                         "seed": derive_seed(master_seed, "select_weights"),
                         "shuffle_seed": derive_seed(master_seed, "select_shuffle")})
    patience = config.get("patience")
    early_stopping = (EarlyStopping(patience=patience,
                                    min_delta=config.get("min_delta") or 0.0)
                      if patience is not None else None)
    history = model.fit(
        X_train, Y_train,
        epochs=config.get("epochs", 100),
        batch_size=config.get("batch_size"),
        validation_data=(X_val, Y_val),
        early_stopping=early_stopping,
        metrics={report_name: report_fn} if report_fn else None,
    )
    epochs_run = len(history[val_key])
    if early_stopping is None:
        selected_idx = epochs_run - 1
    else:
        pick = np.argmax if greater_is_better else np.argmin
        selected_idx = int(pick(history[val_key]))
    # Keep the existing summary keys: without early stopping they now refer
    # to the final epoch, not an earlier validation peak.
    return {
        f"train_{metric}_mean": float(history[train_key][selected_idx]),
        f"train_{metric}_std": None,
        f"val_{metric}_mean": float(history[val_key][selected_idx]),
        f"val_{metric}_std": None,
        "best_epoch_median": selected_idx + 1,
        "best_epochs": [selected_idx + 1],
        "epochs_run": [epochs_run],
    }
