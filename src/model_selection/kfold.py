"""K-fold cross-validation (and simple hold-out).

Estimates validation MEE/MSE for each hyperparameter configuration during model
selection. Returns per-fold metrics so we can report mean +/- std.
"""

from __future__ import annotations

import numpy as np

from ..utils.metrics import METRICS
from .early_stopping import EarlyStopping


def kfold_indices(n_samples: int, k: int, seed: int | None = None):
    """Yield (train_idx, val_idx) for each of the k folds."""
    # Random generator; an optional seed makes the splits reproducible
    # (same seed = same folds on every run).
    rng = np.random.default_rng(seed)

    # Index array (0 .. n_samples - 1).
    indices = np.arange(n_samples)

    # Shuffle before splitting: if the dataset were ordered by class, a fold
    # without shuffling would hold non-representative samples and bias validation.
    rng.shuffle(indices)

    # Split into k sub-arrays (np.array_split also handles a non-divisible n).
    folds = np.array_split(indices, k)

    # Build the train/validation sets for each iteration.
    for i in range(k):
        # The i-th fold is the validation set.
        val_idx = folds[i]

        # All the other folds are concatenated into the training set.
        train_idx = np.concatenate(folds[:i] + folds[i+1:])

        # Yield a (train, val) pair on each `for` iteration.
        yield train_idx, val_idx


def cross_validate(build_model, config: dict, X, Y, k: int = 5, seed: int | None = None, metric: str = "loss") -> dict:
    """Run k-fold CV for one config; return aggregated metrics (mean, std)."""

    # Resolve the metric name to (function, direction, history keys) ONCE, up front.
    if metric == "loss":
        metric_fn, greater_is_better = None, False
        val_key, train_key = "val_loss", "loss"
    else:
        metric_fn, greater_is_better = METRICS[metric]
        val_key, train_key = f"val_{metric}", metric
    pick = np.argmax if greater_is_better else np.argmin

    # Accumulate the per-fold metrics.
    val_scores = []
    train_scores = []
    best_epochs = []

    # Iterate over the k folds produced by kfold_indices.
    for train_idx, val_idx in kfold_indices(len(X), k, seed):

        # Split the data using the indices.
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # Build a fresh model (weights re-initialized from scratch) for this fold.
        model = build_model(config)

        # Training hyperparameters (with defaults).
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', None)

        # Optional early stopping, if requested in the config.
        patience = config.get('patience', None)
        if patience is not None:
            min_delta = config.get('min_delta', 0.0)
            es_callback = EarlyStopping(patience=patience, min_delta=min_delta)
        else:
            es_callback = None

        # train on this fold, with early stopping based on val performance
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            early_stopping=es_callback,
            metrics={metric: metric_fn} if metric_fn else None
        )

        # Pick the epoch with the lowest validation loss, and the training loss
        # at that same epoch.
        best_idx = int(pick(history[val_key]))
        val_scores.append(history[val_key][best_idx])
        train_scores.append(history[train_key][best_idx])
        best_epochs.append(best_idx + 1)

    # Aggregate: mean and std across the k folds (cast to float so the result
    # is JSON-serializable).
    return {
        f"val_{metric}_mean": float(np.mean(val_scores)),
        f"val_{metric}_std": float(np.std(val_scores)),
        f"train_{metric}_mean": float(np.mean(train_scores)),
        f"train_{metric}_std": float(np.std(train_scores)),
        f"best_epoch_median": int(np.median(best_epochs)),
        f"best_epochs": best_epochs
    }
