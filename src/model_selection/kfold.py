"""K-fold cross-validation (and simple hold-out).

Used to estimate validation MEE/MSE for each hyperparameter configuration
during model selection (GUIDA §3.1). Returns per-fold metrics so the report
can show mean +/- std.

To be implemented in F4.
"""

from __future__ import annotations

import numpy as np


def kfold_indices(n_samples: int, k: int, seed: int | None = None):
    """Yield (train_idx, val_idx) for each of the k folds."""
    raise NotImplementedError


def cross_validate(build_model, X, Y, k: int = 5, seed: int | None = None) -> dict:
    """Run k-fold CV for one config; return aggregated metrics (mean, std)."""
    raise NotImplementedError
