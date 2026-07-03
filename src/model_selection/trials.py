"""Multi-seed training + assessment.

Trains one configuration on ``n_trials`` different seeds and aggregates the
scores (mean/std) so results are reported over trials rather than from a single
lucky/unlucky run. Used by both the CLI/API runner (train and assess modes) and
kept independent of any single experiment.

The score to aggregate is injected as ``score_fn``/``score_name`` (accuracy for
MONK, MEE for CUP), so this module never hardcodes a task-specific metric.
"""

from __future__ import annotations

import numpy as np

from .build import build_model
from .early_stopping import EarlyStopping
from ..data.preprocessing import StandardScaler


def run_trials(config: dict, X_tr, y_tr, X_te, y_te, metrics: dict,
               score_fn, score_name: str) -> tuple[dict, list[dict]]:
    """Runs config on n_trials different seeds; returns (summary, histories)."""
    n_trials  = config.get("n_trials", 5)
    base_seed = config.get("seed", 0)

    test_score, train_score, histories = [], [], []
    for i in range(n_trials):
        cfg_i = {**config, "seed": base_seed + i}
        
        # Facciamo una copia locale dei dati in modo che i fold/trial siano indipendenti
        X_tr_i, y_tr_i = np.copy(X_tr), np.copy(y_tr)
        X_te_i, y_te_i = np.copy(X_te), np.copy(y_te)
        scaler_Y = None
        
        if config.get("scale", False):
            scaler_X = StandardScaler()
            X_tr_i = scaler_X.fit_transform(X_tr_i)
            X_te_i = scaler_X.transform(X_te_i)
            
            scaler_Y = StandardScaler()
            y_tr_i = scaler_Y.fit_transform(y_tr_i)
            y_te_i = scaler_Y.transform(y_te_i)

        model = build_model(cfg_i)

        # Opt-in, like cross_validate: no "patience" in config => no early
        # stopping (train for the full epoch count requested).
        patience = cfg_i.get("patience", None)
        es = (EarlyStopping(patience=patience, min_delta=cfg_i.get("min_delta", 0.0))
              if patience is not None else None)

        history = model.fit(X_tr_i, y_tr_i, epochs=cfg_i["epochs"],
                            batch_size=cfg_i.get("batch_size"),
                            validation_data=(X_te_i, y_te_i), metrics=metrics,
                            early_stopping=es)

        pred_tr, pred_te = model.predict(X_tr_i), model.predict(X_te_i)
        
        # Se abbiamo scalato le Y, per calcolare il vero score_fn (MEE) dobbiamo 
        # invertire la trasformazione per tornare alla scala originale.
        if scaler_Y is not None:
            pred_tr = scaler_Y.inverse_transform(pred_tr)
            pred_te = scaler_Y.inverse_transform(pred_te)
            y_tr_curr = scaler_Y.inverse_transform(y_tr_i)
            y_te_curr = scaler_Y.inverse_transform(y_te_i)
        else:
            y_tr_curr, y_te_curr = y_tr_i, y_te_i

        s_tr, s_te = score_fn(pred_tr, y_tr_curr), score_fn(pred_te, y_te_curr)
        train_score.append(s_tr); test_score.append(s_te); histories.append(history)

    test_score, train_score = np.array(test_score), np.array(train_score)
    rep = int(np.argsort(test_score)[len(test_score) // 2])

    summary = {
        f"test_{score_name}_mean":  float(np.mean(test_score)),
        f"test_{score_name}_std":   float(np.std(test_score)),
        f"train_{score_name}_mean": float(np.mean(train_score)),
        f"train_{score_name}_std":  float(np.std(train_score)),
        f"per_trial_{score_name}": [float(a) for a in test_score],
        "n_trials": n_trials,
        "representative_trial": rep,
    }
    return summary, histories
