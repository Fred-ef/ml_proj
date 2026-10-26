"""Scale a training split and score scaled targets in the original units"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..data.preprocessing import StandardScaler


def prepare_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray | None,
    y_eval: np.ndarray | None,
    *,
    scale: bool,
    classification: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None,
           StandardScaler | None]:
    """Fit the scalers on the training data only; MONK data is never scaled"""
    if classification and scale:
        raise ValueError("scaling MONK data is unsupported; use scale=False")
    if not scale:
        return X_train, y_train, X_eval, y_eval, None

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    if X_eval is not None:
        X_eval = scaler_X.transform(X_eval)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    if y_eval is not None:
        y_eval = scaler_y.transform(y_eval)
    return X_train, y_train, X_eval, y_eval, scaler_y


def original_target_metric(metric_fn: Callable, scaler_y: StandardScaler | None) -> Callable:
    """Wrap metric_fn so that it scores in the original target units"""
    if scaler_y is None:
        return metric_fn

    def score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return metric_fn(scaler_y.inverse_transform(y_pred),
                         scaler_y.inverse_transform(y_true))

    return score
