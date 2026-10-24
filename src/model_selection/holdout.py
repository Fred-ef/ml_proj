"""Evaluate one configuration on a single, shared training/validation split."""

from __future__ import annotations

import numpy as np

from ..utils.metrics import METRICS
from ..utils.seeding import derive_seed, resolve_seed
from .early_stopping import EarlyStopping
from .scaling import original_target_metric, prepare_split


def holdout_indices(n_samples: int, val_fraction: float, seed: int | None = None,
                    *, stratify_labels=None):
    """Return train/validation indices, optionally preserving class proportions."""
    if n_samples < 2:
        raise ValueError("holdout requires at least two samples")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    n_val = min(max(1, int(np.ceil(n_samples * val_fraction))), n_samples - 1)
    rng = np.random.default_rng(seed)
    if stratify_labels is None:
        indices = rng.permutation(n_samples)
        return indices[:-n_val], indices[-n_val:]

    labels = np.asarray(stratify_labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1 or len(labels) != n_samples:
        raise ValueError("stratify_labels must have shape (n_samples,) or (n_samples, 1)")

    classes, class_ids, counts = np.unique(labels, return_inverse=True,
                                            return_counts=True)
    n_classes = len(classes)
    if np.any(counts < 2):
        raise ValueError("stratified holdout requires at least two samples per class")
    if n_val < n_classes or n_samples - n_val < n_classes:
        raise ValueError("stratified holdout needs room for every class in both splits")

    # Allocate exactly n_val samples. Start near the proportional quotas, then
    # correct rounding while keeping at least one sample of each class per split.
    quotas = counts * (n_val / n_samples)
    val_counts = np.clip(np.floor(quotas).astype(int), 1, counts - 1)
    while val_counts.sum() < n_val:
        eligible = val_counts < counts - 1
        chosen = int(np.argmax(np.where(eligible, quotas - val_counts, -np.inf)))
        val_counts[chosen] += 1
    while val_counts.sum() > n_val:
        eligible = val_counts > 1
        chosen = int(np.argmax(np.where(eligible, val_counts - quotas, -np.inf)))
        val_counts[chosen] -= 1

    train_parts, val_parts = [], []
    for class_id, count in enumerate(val_counts):
        indices = rng.permutation(np.flatnonzero(class_ids == class_id))
        val_parts.append(indices[:count])
        train_parts.append(indices[count:])
    return (rng.permutation(np.concatenate(train_parts)),
            rng.permutation(np.concatenate(val_parts)))


def holdout_validate(build_model, config: dict, X, Y, train_idx, val_idx,
                     seed: int | None = None, metric: str = "loss", trial: int = 0) -> dict:
    """Train once; use the final epoch unless early stopping is enabled."""
    if metric == "loss":
        metric_fn, greater_is_better = None, False
        train_key, val_key = "loss", "val_loss"
    else:
        metric_fn, greater_is_better = METRICS[metric]
        train_key, val_key = metric, f"val_{metric}"

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_train, Y_train, X_val, Y_val, scaler_Y = prepare_split(
        X_train, Y_train, X_val, Y_val,
        scale=bool(config.get("scale")),
        classification=metric == "acc" or str(config.get("task", "")).startswith("monk"))

    report_fn = metric_fn
    report_name = metric if metric_fn else None
    if scaler_Y is not None:
        # Model.fit operates on scaled targets. Report model-selection errors in
        # the original units so configurations with and without scaling remain
        # comparable. The optimizer still uses its ordinary training loss.
        original_fn = metric_fn if metric_fn else METRICS["mse"][0]
        report_fn = original_target_metric(original_fn, scaler_Y)
        if metric == "loss":
            report_name = "reported_loss"
            train_key, val_key = report_name, f"val_{report_name}"
        else:
            report_name = metric

    master_seed = resolve_seed(seed)
    model = build_model({**config,
                         "seed": derive_seed(master_seed, "select_weights", trial),
                         "shuffle_seed": derive_seed(master_seed, "select_shuffle", trial)})
    patience = config.get("patience")
    early_stopping = (EarlyStopping(patience=patience,
                                    min_delta=config.get("min_delta") or 0.0,
                                    monitor="val_loss" if metric == "acc" else val_key,
                                    mode="min" if metric == "acc" else
                                    "max" if greater_is_better else "min")
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
        pick = np.argmax if early_stopping.mode == "max" else np.argmin
        selected_idx = int(pick(history[early_stopping.monitor]))
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
        "train_scores": [history[train_key][selected_idx]],
        "val_scores": [history[val_key][selected_idx]],
        "train_curves": [history[train_key]],
        "val_curves": [history[val_key]],
    }
