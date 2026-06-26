"""K-fold cross-validation (and simple hold-out).

Estimates validation MEE/MSE for each hyperparameter configuration during model
selection. Returns per-fold metrics so we can report mean +/- std.
"""

from __future__ import annotations

import numpy as np

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


def cross_validate(build_model, config: dict, X, Y, k: int = 5, seed: int | None = None) -> dict:
    """Run k-fold CV for one config; return aggregated metrics (mean, std)."""
    # Accumulate the per-fold metrics.
    val_scores = []
    train_scores = []

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

        # Train on this fold; pass the validation data so the metric is computed
        # every epoch.
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            early_stopping=es_callback
        )

        # Pick the epoch with the lowest validation loss, and the training loss
        # at that same epoch.
        best_idx = int(np.argmin(history['val_loss']))

        best_val_loss = history['val_loss'][best_idx]
        best_train_loss = history['loss'][best_idx]

        # Store this fold's results.
        val_scores.append(best_val_loss)
        train_scores.append(best_train_loss)

    # Aggregate: mean and std across the k folds (cast to float so the result
    # is JSON-serializable).
    return {
        'val_mee_mean': float(np.mean(val_scores)),
        'val_mee_std': float(np.std(val_scores)),
        'train_mee_mean': float(np.mean(train_scores)),
        'train_mee_std': float(np.std(train_scores))
    }
