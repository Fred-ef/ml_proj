"""Train and save a CUP model on the full labelled training set.

The assessment run supplies the chosen configuration and the seeds of trial 0.
Its internal test set is not consulted during this final fit. Prediction on the
blind test and submission CSV export use ``experiments.cup_predict``.

Usage::

    python -m experiments.cup_final --assessment-run results/cup/RUN_ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.data.cup import N_INPUTS, N_TARGETS, load_cup_train
from src.data.preprocessing import StandardScaler
from src.model_selection.build import build_model
from src.model_selection.scaling import original_target_metric
from src.utils.experiment import new_run_dir, save_run
from src.utils.metrics import mee
from src.utils.model_artifact import save_model_artifact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CSV = ROOT / "data" / "cup" / "ML-CUP25-TR.csv"


def _saved_assessment(assessment_run: Path) -> tuple[dict, dict, dict]:
    """Read a completed CUP assessment and its precommitted trial-0 seeds."""
    config = json.loads((assessment_run / "config.json").read_text(encoding="utf-8"))["config"]
    summary = json.loads((assessment_run / "summary.json").read_text(encoding="utf-8"))
    if config.get("task") != "cup" or config.get("mode") != "assess":
        raise ValueError("assessment_run must contain a completed CUP assess run")
    if "mee_ts_mean" not in summary:
        raise ValueError("assessment_run has no CUP test assessment summary")

    seed_plan = summary.get("seed_plan", {})
    trials = seed_plan.get("trials", [])
    if not trials:
        raise ValueError("assessment_run has no recorded trial seeds")
    if seed_plan.get("root") != config.get("seed"):
        raise ValueError("assessment seed plan does not match its saved config")
    trial_zero = trials[0]
    for key in ("weights", "shuffle"):
        seed = trial_zero.get(key)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"assessment trial 0 has no valid {key} seed")
    return config, summary, trial_zero


def _final_config(assessment_config: dict, trial_zero: dict) -> dict:
    """Keep fitted model settings, replacing only assessment bookkeeping."""
    omit = {"mode", "tag", "n_trials", "seed_source", "internal_test_split",
            "val_mean", "val_std", "patience"}
    config = {k: v for k, v in assessment_config.items() if k not in omit}
    config.update({"task": "cup", "seed": trial_zero["weights"],
                   "shuffle_seed": trial_zero["shuffle"]})
    if config.get("n_inputs") != N_INPUTS:
        raise ValueError(f"CUP model must have {N_INPUTS} inputs")
    arch = config.get("arch")
    if not isinstance(arch, list) or not arch or arch[-1].get("units") != N_TARGETS:
        raise ValueError(f"CUP model must have {N_TARGETS} outputs")
    if arch[-1].get("act") != "identity":
        raise ValueError("CUP output activation must be identity")
    epochs = config.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("assessment epochs must be a positive integer")

    target = config.get("cv_stop_target")
    if target is not None:
        if (isinstance(target, bool) or not isinstance(target, (int, float))
                or not math.isfinite(target) or target < 0):
            raise ValueError("cv_stop_target must be finite and non-negative")
        if config.get("cv_stop_metric", "mee") != "mee":
            raise ValueError("CUP cv_stop_metric must be 'mee'")
    elif config.get("cv_stop_metric") is not None:
        raise ValueError("cv_stop_metric requires cv_stop_target")
    return config


def train_final_cup(
    assessment_run: str | Path,
    *,
    training_csv: str | Path = DEFAULT_TRAINING_CSV,
    results_root: str | Path | None = None,
) -> Path:
    """Refit assessment trial 0 on every labelled CUP row and save the model.

    The trial-0 weight and shuffle seeds are fixed by the saved assessment, but
    its 400-row weights are not reused. This fit starts from the same initial
    weights and trains on all available labelled rows, including the former
    internal test partition. No test result is used to choose the seed.
    """
    assessment_run = Path(assessment_run).expanduser().resolve()
    training_csv = Path(training_csv).expanduser().resolve()
    source_config, source_summary, trial_zero = _saved_assessment(assessment_run)
    config = _final_config(source_config, trial_zero)

    ids, X, y = load_cup_train(str(training_csv))
    if (X.ndim != 2 or y.ndim != 2 or X.shape[1] != N_INPUTS
            or y.shape != (len(X), N_TARGETS) or len(X) == 0):
        raise ValueError(f"CUP training CSV must contain id, {N_INPUTS} inputs, "
                         f"and {N_TARGETS} targets per row")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("CUP training CSV contains duplicate IDs or non-finite values")

    scaler_x = scaler_y = None
    X_fit, y_fit = X, y
    if config.get("scale"):
        scaler_x, scaler_y = StandardScaler(), StandardScaler()
        X_fit = scaler_x.fit_transform(X)
        y_fit = scaler_y.fit_transform(y)

    model = build_model(config)
    target = config.get("cv_stop_target")
    history = model.fit(
        X_fit, y_fit, epochs=config["epochs"], batch_size=config.get("batch_size"),
        metrics={"mee": original_target_metric(mee, scaler_y)},
        training_target=("mee", target) if target is not None else None,
    )

    training_hash = hashlib.sha256(training_csv.read_bytes()).hexdigest()
    summary = {
        "training_rows": len(X),
        "train_mee": float(history["mee"][-1]),
        "epochs_run": model.epochs_run,
        "cv_stop_target": target,
        "cv_stop_reached": None if target is None else bool(history["mee"][-1] <= target),
        "selected_trial": 0,
        "source_assessment_run": assessment_run.name,
        "source_assessment_seed": source_config["seed"],
        "training_data_sha256": training_hash,
    }
    metadata = {
        **summary,
        "training_csv": str(training_csv),
        "seed_plan_scheme": source_summary["seed_plan"].get("scheme"),
    }

    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    run_dir = new_run_dir(results_root, "cup", "cup-final", trial_zero["weights"])
    save_model_artifact(run_dir / "model.npz", model, config, scaler_x, scaler_y,
                        metadata=metadata)
    save_run(run_dir, config, summary, history=history)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a final CUP model from an assess run.")
    parser.add_argument("--assessment-run", required=True,
                        help="results/cup/RUN_ID from a completed assess run")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAINING_CSV),
                        help="labelled CUP CSV (defaults to data/cup/ML-CUP25-TR.csv)")
    parser.add_argument("--results-root", help="directory for the final run")
    args = parser.parse_args(argv)
    run_dir = train_final_cup(args.assessment_run, training_csv=args.train_data,
                              results_root=args.results_root)
    print(f"Final CUP model saved: {run_dir / 'model.npz'}")


if __name__ == "__main__":
    main()
