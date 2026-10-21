"""Aggregate per-epoch curves of runs that may stop at different epochs."""

from __future__ import annotations

import warnings

import numpy as np


def pad_curves(runs: list[list[list[float]]]) -> np.ndarray:
    """[T][F] curves of any length -> array (T, F, E_max), NaN after a run stops."""
    n_trials, n_folds = len(runs), len(runs[0])
    e_max = max(len(curve) for trial in runs for curve in trial)
    out = np.full((n_trials, n_folds, e_max), np.nan)
    for t, trial in enumerate(runs):
        for f, curve in enumerate(trial):
            out[t, f, :len(curve)] = curve
    return out


def aggregate_curves(runs: list[list[list[float]]]) -> dict:
    """Epoch-by-epoch mean and std across initializations (trials)."""
    A = pad_curves(runs)                                  # (T, F, E)
    with warnings.catch_warnings():
        # A trial whose folds have all stopped yields an all-NaN slice: expected.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_trial = np.nanmean(A, axis=1)                 # (T, E) fold average
        mean = np.nanmean(per_trial, axis=0)              # (E,)
        std = np.nanstd(per_trial, axis=0)                # (E,)
    alive = np.sum(~np.isnan(A), axis=(0, 1))            # (E,) runs still training
    return {"mean": mean.tolist(), "std": std.tolist(), "n_alive": alive.tolist()}
