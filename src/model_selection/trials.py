"""Repeated training of one configuration (train and assess modes)

The configuration is trained with ``n_trials`` seeds and the scores are reported
as mean and std over the trials. The score is passed as ``score_fn``/``score_name``
(accuracy for MONK, MEE for CUP)
"""

from __future__ import annotations

import numpy as np

from .build import build_model
from .early_stopping import EarlyStopping
from .holdout import holdout_indices
from .scaling import original_target_metric, prepare_split
from ..utils.metrics import METRICS
from ..utils.seeding import SEED_SCHEME, derive_seed, resolve_seed


def run_trials(config: dict, X_tr, y_tr, X_te, y_te, metrics: dict,
               score_fn, score_name: str, *, stage: str = "assess") -> tuple[dict, list[dict]]:
    """Train n_trials times; only the assess stage scores on the test set"""
    if stage not in ("train", "assess"):
        raise ValueError(f"unknown trial stage: {stage!r}")
    n_trials  = config.get("n_trials", 5)
    master_seed = resolve_seed(config.get("seed", 0))
    patience = config.get("patience") if stage == "train" else None
    target = config.get("cv_stop_target") if stage == "assess" else None
    target_metric = "loss" if score_name == "acc" else score_name
    stratify = bool(config.get("stratify", False))
    if stratify and stage != "train":
        raise ValueError("stratify applies only to train validation")
    if stratify and score_name != "acc":
        raise ValueError("stratify requires a classification task")
    use_validation = stage == "train" and (patience is not None or
                                             config.get("val_fraction") is not None or stratify)
    split_seed = derive_seed(master_seed, "train_split") if use_validation else None
    val_fraction = config.get("val_fraction")
    train_idx, val_idx = (holdout_indices(len(X_tr), 0.2 if val_fraction is None else val_fraction,
                                         seed=split_seed,
                                         **({"stratify_labels": y_tr} if stratify else {}))
                          if use_validation else
                          (None, None))
    seed_plan = {"scheme": SEED_SCHEME, "root": master_seed, "trials": [
        {"weights": derive_seed(master_seed, "trial_weights", i),
         "shuffle": derive_seed(master_seed, "trial_shuffle", i)}
        for i in range(n_trials)
    ]}
    if use_validation:
        seed_plan["split"] = split_seed

    eval_score, train_score, histories = [], [], []
    selected_epochs, epochs_run = [], []
    target_reached, target_final_errors = [], []
    for i in range(n_trials):
        cfg_i = {**config, "seed": seed_plan["trials"][i]["weights"],
                 "shuffle_seed": seed_plan["trials"][i]["shuffle"]}
        
        # Facciamo una copia locale dei dati in modo che i fold/trial siano indipendenti
        X_tr_i = np.copy(X_tr if train_idx is None else X_tr[train_idx])
        y_tr_i = np.copy(y_tr if train_idx is None else y_tr[train_idx])
        X_ev_i = (np.copy(X_te) if stage == "assess" else
                  np.copy(X_tr[val_idx]) if use_validation else None)
        y_ev_i = (np.copy(y_te) if stage == "assess" else
                  np.copy(y_tr[val_idx]) if use_validation else None)
        X_tr_i, y_tr_i, X_ev_i, y_ev_i, scaler_Y = prepare_split(
            X_tr_i, y_tr_i, X_ev_i, y_ev_i,
            scale=bool(config.get("scale")),
            classification=score_name == "acc" or str(config.get("task", "")).startswith("monk"))

        # The loss uses scaled targets, the reported metrics the original units
        # (also used by CUP's training target)
        fit_metrics = {name: original_target_metric(fn, scaler_Y)
                       for name, fn in metrics.items()}
        if use_validation or target is not None:
            fit_metrics[score_name] = original_target_metric(score_fn, scaler_Y)

        model = build_model(cfg_i)

        # Early stopping only in train mode; assess can instead stop when the
        # training error on the whole design set reaches the CV target
        es = (EarlyStopping(patience=patience,
                            min_delta=cfg_i.get("min_delta") or 0.0,
                            monitor="val_loss" if score_name == "acc" else f"val_{score_name}",
                            mode="min" if score_name == "acc" else
                            "max" if METRICS.get(score_name, (None, False))[1]
                            else "min")
              if patience is not None else None)

        # The test curve is only recorded: the target checks a training metric,
        # so no stopping decision depends on the test set
        fit_kwargs = dict(epochs=cfg_i["epochs"], batch_size=cfg_i.get("batch_size"),
                          validation_data=((X_ev_i, y_ev_i) if X_ev_i is not None else None),
                          metrics=fit_metrics, early_stopping=es)
        if target is not None:
            fit_kwargs["training_target"] = (target_metric, target)
        history = model.fit(X_tr_i, y_tr_i, **fit_kwargs)

        if es is not None:
            selected_epochs.append(es.best_epoch)
            epochs_run.append(len(history["loss"]))
        if target is not None:
            error = float(history[target_metric][-1])
            epochs_run.append(len(history["loss"]))
            target_final_errors.append(error)
            target_reached.append(error <= target)

        pred_tr = model.predict(X_tr_i)
        pred_ev = model.predict(X_ev_i) if X_ev_i is not None else None
        
        # Se abbiamo scalato le Y, per calcolare il vero score_fn (MEE) dobbiamo 
        # invertire la trasformazione per tornare alla scala originale
        if scaler_Y is not None:
            pred_tr = scaler_Y.inverse_transform(pred_tr)
            if pred_ev is not None:
                pred_ev = scaler_Y.inverse_transform(pred_ev)
            y_tr_curr = scaler_Y.inverse_transform(y_tr_i)
            y_ev_curr = scaler_Y.inverse_transform(y_ev_i) if y_ev_i is not None else None
        else:
            y_tr_curr, y_ev_curr = y_tr_i, y_ev_i

        train_score.append(score_fn(pred_tr, y_tr_curr))
        if pred_ev is not None:
            eval_score.append(score_fn(pred_ev, y_ev_curr))
        histories.append(history)

    eval_score, train_score = np.array(eval_score), np.array(train_score)
    representative_scores = eval_score if len(eval_score) else train_score
    rep = int(np.argsort(representative_scores)[len(representative_scores) // 2])

    summary = {
        f"train_{score_name}_mean": float(np.mean(train_score)),
        f"train_{score_name}_std":  float(np.std(train_score)),
        "n_trials": n_trials,
        "representative_trial": rep,
        "seed_plan": seed_plan,
    }
    if len(eval_score):
        label = "test" if stage == "assess" else "val"
        summary.update({
            f"{label}_{score_name}_mean": float(np.mean(eval_score)),
            f"{label}_{score_name}_std": float(np.std(eval_score)),
            (f"per_trial_{score_name}" if stage == "assess" else
             f"per_trial_val_{score_name}"): [float(a) for a in eval_score],
        })
    else:
        summary[f"per_trial_train_{score_name}"] = [float(a) for a in train_score]
    if stage == "train" and patience is not None:
        summary["selected_epochs"] = selected_epochs
        summary["epochs_run"] = epochs_run
    if target is not None:
        summary.update({"cv_stop_target": float(target),
                        "cv_stop_metric": target_metric,
                        "cv_stop_reached": target_reached,
                        "cv_stop_final_errors": target_final_errors,
                        "epochs_run": epochs_run})
    return summary, histories
