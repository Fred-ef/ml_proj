"""K-fold cross-validation (and simple hold-out).

Estimates validation MEE/MSE for each hyperparameter configuration during model
selection. Returns per-fold metrics so we can report mean +/- std.
"""

from __future__ import annotations

import numpy as np

from ..utils.metrics import METRICS
from .early_stopping import EarlyStopping
from ..data.preprocessing import StandardScaler


def _strata(labels, n_samples: int):
    """Return class indices for single-output classification targets."""
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1 or len(labels) != n_samples:
        raise ValueError("stratification requires one label per sample")
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
        raise ValueError("stratification requires finite discrete class labels")
    classes, inverse, counts = np.unique(labels, return_inverse=True,
                                          return_counts=True)
    if len(classes) < 2:
        raise ValueError("stratification requires at least two classes")
    return inverse, counts


def kfold_indices(n_samples: int, k: int, seed: int | None = None, labels=None):
    """Yield (train_idx, val_idx) for each of the k folds."""
    if k < 2:
        raise ValueError("k must be at least 2")
    if k > n_samples:
        raise ValueError(f"k={k} cannot exceed n_samples={n_samples}")
    # Use a seed when reproducible folds matter.
    rng = np.random.default_rng(seed)

    if labels is None:
        indices = rng.permutation(n_samples)
        folds = np.array_split(indices, k)
    else:
        inverse, counts = _strata(labels, n_samples)
        if np.any(counts < k):
            raise ValueError("each class needs at least k samples for stratified k-fold")
        per_class = [np.array_split(rng.permutation(np.flatnonzero(inverse == cls)), k)
                     for cls in range(len(counts))]
        folds = [rng.permutation(np.concatenate([parts[i] for parts in per_class]))
                 for i in range(k)]

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        yield train_idx, val_idx


def holdout_indices(n_samples: int, validation_fraction: float,
                    seed: int | None = None, labels=None):
    """One reproducible train/validation split of the design set."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if n_samples < 2:
        raise ValueError("holdout selection requires at least two samples")
    n_val = min(n_samples - 1, max(1, int(round(n_samples * validation_fraction))))
    rng = np.random.default_rng(seed)
    if labels is None:
        indices = rng.permutation(n_samples)
        return indices[n_val:], indices[:n_val]

    inverse, counts = _strata(labels, n_samples)
    n_classes = len(counts)
    if np.any(counts < 2) or n_val < n_classes or n_samples - n_val < n_classes:
        raise ValueError("stratified holdout needs each class in train and validation")

    # Match class proportions while reserving at least one training example
    # from every class and keeping the requested validation size exact.
    ideal = counts * n_val / n_samples
    val_counts = np.ones(n_classes, dtype=int)
    for _ in range(n_val - n_classes):
        eligible = np.flatnonzero(val_counts < counts - 1)
        chosen = eligible[np.argmax(ideal[eligible] - val_counts[eligible])]
        val_counts[chosen] += 1

    train_parts, val_parts = [], []
    for cls, n_cls_val in enumerate(val_counts):
        indices = rng.permutation(np.flatnonzero(inverse == cls))
        val_parts.append(indices[:n_cls_val])
        train_parts.append(indices[n_cls_val:])
    return (rng.permutation(np.concatenate(train_parts)),
            rng.permutation(np.concatenate(val_parts)))


def cross_validate(build_model, config: dict, X, Y, k: int = 5, seed: int | None = None,
                   metric: str = "loss", split_seed: int | None = None,
                   stratify: bool = False) -> dict:
    """Run k-fold CV for one config; return aggregated metrics (mean, std)."""

    return _validate_splits(build_model, config, X, Y,
                            kfold_indices(len(X), k, seed if split_seed is None else split_seed,
                                          labels=Y if stratify else None),
                            seed, metric)


def holdout_validate(build_model, config: dict, X, Y, validation_fraction: float,
                     seed: int | None = None, metric: str = "loss",
                     split_seed: int | None = None, stratify: bool = False) -> dict:
    """Score one config on one held-out slice of the design set."""

    split = holdout_indices(len(X), validation_fraction,
                            seed if split_seed is None else split_seed,
                            labels=Y if stratify else None)
    return _validate_splits(build_model, config, X, Y, [split], seed, metric)


def _validate_splits(build_model, config: dict, X, Y, splits, seed, metric) -> dict:
    """Shared fitting and metric selection for k-fold and single holdout."""

    if len(X) != len(Y) or len(X) == 0:
        raise ValueError("X and Y must contain the same non-zero number of rows")
    if config.get("epochs", 100) <= 0:
        raise ValueError("epochs must be greater than zero")

    # Resolve the metric once before looping over folds.
    if metric == "loss":
        metric_fn, greater_is_better = None, False
        val_key, train_key = "val_loss", "loss"
    else:
        metric_fn, greater_is_better = METRICS[metric]
        val_key, train_key = f"val_{metric}", metric
    pick = np.argmax if greater_is_better else np.argmin

    val_scores = []
    train_scores = []
    best_epochs = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        if config.get("scale", False):
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_val = scaler_X.transform(X_val)
            
            scaler_Y = StandardScaler()
            Y_train = scaler_Y.fit_transform(Y_train)
            Y_val = scaler_Y.transform(Y_val)

            if metric_fn is not None:
                def unscaled_metric(y_pred, y_true, base_fn=metric_fn, sy=scaler_Y):
                    yp = sy.inverse_transform(y_pred)
                    yt = sy.inverse_transform(y_true)
                    return base_fn(yp, yt)
                fold_metrics = {metric: unscaled_metric}
            else:
                fold_metrics = None
        else:
            fold_metrics = {metric: metric_fn} if metric_fn else None

        # Give each fold its own reproducible model seed.
        fold_seed = None if seed is None else seed + fold_idx
        model = build_model({**config, "seed": fold_seed})

        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', None)

        patience = config.get('patience', None)
        if patience is not None:
            min_delta = config.get('min_delta', 0.0)
            monitor = config.get('monitor', 'val_loss')
            if monitor == "val_loss":
                monitor_mode = "min"
            elif metric != "loss" and monitor == f"val_{metric}":
                monitor_mode = "max" if greater_is_better else "min"
            else:
                raise ValueError(
                    f"monitor={monitor!r} is unavailable; use 'val_loss' or {val_key!r}"
                )
            es_callback = EarlyStopping(
                patience=patience,
                min_delta=min_delta,
                monitor=monitor,
                mode=monitor_mode,
            )
        else:
            es_callback = None

        # Train this fold, with optional early stopping on validation performance.
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            early_stopping=es_callback,
            metrics=fold_metrics
        )

        # Use the training and validation scores from the same best epoch.
        best_idx = int(pick(history[val_key]))
        val_scores.append(history[val_key][best_idx])
        train_scores.append(history[train_key][best_idx])
        best_epochs.append(best_idx + 1)

    # Return plain Python values so the result is easy to serialize.
    return {
        f"val_{metric}_mean": float(np.mean(val_scores)),
        f"val_{metric}_std": float(np.std(val_scores)) if len(val_scores) > 1 else None,
        f"train_{metric}_mean": float(np.mean(train_scores)),
        f"train_{metric}_std": float(np.std(train_scores)) if len(train_scores) > 1 else None,
        f"best_epoch_median": int(np.median(best_epochs)),
        f"best_epochs": best_epochs
    }
