"""Entry point: retrain the selected CUP model and produce the blind-test output.

Steps (after the selected model has been independently assessed):
    1. Retrain the chosen model on all training data.
    2. Predict on the 1000 blind-test patterns.
    3. Write ``<team-name>_ML-CUP25-TS.csv`` in the required format:
         - 4 comment lines (# names / # team nickname / # ML-CUP25 v1 / # date)
         - 1000 rows "id,o1,o2,o3,o4", no spaces after commas, ids 1..1000.
    4. Write ``<team-name>_abstract.txt`` (~5 lines).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np

from src.data.cup import N_INPUTS, N_TARGETS, load_cup_blind, load_cup_train
from src.data.preprocessing import StandardScaler
from src.model_selection.build import build_model

ROOT = Path(__file__).resolve().parents[1]


def _load_assessment_config(path: str | Path) -> dict:
    """Load either a runner envelope or a plain fixed-epoch model config."""

    raw = json.loads(Path(path).read_text())
    if raw.get("task", "cup") != "cup":
        raise ValueError("the final submission config must target task='cup'")
    cfg = dict(raw.get("config", raw))
    forbidden = sorted({"patience", "min_delta", "monitor"} & cfg.keys())
    if forbidden:
        raise ValueError(
            f"final retraining must use fixed epochs; remove early-stopping fields {forbidden}"
        )
    arch = cfg.get("arch")
    if not arch or arch[-1].get("units") != N_TARGETS:
        raise ValueError(f"the CUP output layer must contain exactly {N_TARGETS} units")
    if not isinstance(cfg.get("epochs"), int) or cfg["epochs"] <= 0:
        raise ValueError("epochs must be the positive fixed count selected by cross-validation")
    return cfg


def train_final_model(config: dict, X: np.ndarray, Y: np.ndarray,
                      X_blind: np.ndarray) -> np.ndarray:
    """Retrain once on every labelled CUP row and predict the blind set."""

    X_fit, Y_fit, X_out = np.copy(X), np.copy(Y), np.copy(X_blind)
    y_scaler = None
    if config.get("scale", False):
        x_scaler = StandardScaler()
        X_fit = x_scaler.fit_transform(X_fit)
        X_out = x_scaler.transform(X_out)
        y_scaler = StandardScaler()
        Y_fit = y_scaler.fit_transform(Y_fit)

    model_cfg = {**config, "n_inputs": N_INPUTS}
    model = build_model(model_cfg)
    model.fit(
        X_fit,
        Y_fit,
        epochs=model_cfg["epochs"],
        batch_size=model_cfg.get("batch_size"),
    )
    predictions = model.predict(X_out)
    if y_scaler is not None:
        predictions = y_scaler.inverse_transform(predictions)
    if predictions.shape != (len(X_blind), N_TARGETS):
        raise RuntimeError(f"unexpected prediction shape: {predictions.shape}")
    return predictions


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not stem:
        raise ValueError("team nickname must contain at least one filename-safe character")
    return stem


def write_submission(path: str | Path, names: str, nickname: str, submission_date: str,
                     ids: np.ndarray, predictions: np.ndarray) -> Path:
    """Write the official four-comment-line CUP submission format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(ids) != len(predictions):
        raise ValueError("blind ids and predictions have different row counts")
    with path.open("w", newline="") as handle:
        handle.write(f"# {names}\n# {nickname}\n# ML-CUP25 v1\n# {submission_date}\n")
        writer = csv.writer(handle, lineterminator="\n")
        for sample_id, outputs in zip(ids, predictions):
            writer.writerow([int(sample_id), *(f"{float(value):.10g}" for value in outputs)])
    return path


def _write_abstract(path: Path, config: dict, names: str, nickname: str) -> None:
    arch = "-".join([str(N_INPUTS), *(str(layer["units"]) for layer in config["arch"])])
    optim = config["optim"]
    lines = [
        f"Team {nickname}: {names}.",
        f"The submitted model is a multilayer perceptron with architecture {arch}.",
        f"It was trained with {optim['type']} and {config.get('loss', 'mse')} loss for "
        f"{config['epochs']} fixed epochs selected by cross-validation.",
        f"Input and target standardization was {'used' if config.get('scale', False) else 'not used'}; "
        "all preprocessing statistics were fitted on labelled training data only.",
        "After independent internal-test assessment, the selected model was retrained on all labelled "
        "CUP patterns and used once to predict the blind test set.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the selected CUP model and create submission files")
    parser.add_argument("--config", required=True, help="fixed-epoch CUP assessment JSON")
    parser.add_argument("--team-name", required=True, help="team members/names for the first comment line")
    parser.add_argument("--team-nickname", required=True, help="team nickname and output filename prefix")
    parser.add_argument("--date", dest="submission_date", help="submission date (default: today, DD/MM/YYYY)")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "cup"), help="directory containing CUP CSVs")
    parser.add_argument("--output-dir", default=str(ROOT / "submissions"), help="destination directory")
    args = parser.parse_args()

    config = _load_assessment_config(args.config)
    data_dir = Path(args.data_dir)
    _ids_train, X, Y = load_cup_train(str(data_dir / "ML-CUP25-TR.csv"))
    blind_ids, X_blind = load_cup_blind(str(data_dir / "ML-CUP25-TS.csv"))
    if len(X) != 500 or len(X_blind) != 1000:
        raise ValueError(
            f"unexpected CUP dataset sizes: {len(X)} labelled and {len(X_blind)} blind rows"
        )
    predictions = train_final_model(config, X, Y, X_blind)

    if args.submission_date:
        datetime.strptime(args.submission_date, "%d/%m/%Y")
        submission_date = args.submission_date
    else:
        submission_date = date.today().strftime("%d/%m/%Y")

    stem = _safe_stem(args.team_nickname)
    output_dir = Path(args.output_dir)
    csv_path = write_submission(
        output_dir / f"{stem}_ML-CUP25-TS.csv",
        args.team_name,
        args.team_nickname,
        submission_date,
        blind_ids,
        predictions,
    )
    abstract_path = output_dir / f"{stem}_abstract.txt"
    _write_abstract(abstract_path, config, args.team_name, args.team_nickname)
    print(f"Submission written to {csv_path}")
    print(f"Abstract written to {abstract_path}")


if __name__ == "__main__":
    main()
