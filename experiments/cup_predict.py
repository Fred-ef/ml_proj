"""Load a saved final CUP model and export predictions in submission format.

Usage::

    python -m experiments.cup_predict \
        --model results/cup/FINAL_RUN/model.npz \
        --input data/cup/ML-CUP25-TS.csv \
        --output results/cup/ML-CUP25-submission.csv \
        --members "Name Surname, Name Surname" --team "Team Name" \
        --date 26/10/2026
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.cup import N_INPUTS, N_TARGETS, load_cup_blind
from src.utils.model_artifact import load_model_artifact


def _header_value(value: str, label: str) -> str:
    value = value.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a nonempty single line")
    return value


def _submission_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%d/%m/%Y")
    except ValueError as exc:
        raise ValueError("date must have format DD/MM/YYYY") from exc
    return parsed.strftime("%d/%m/%Y")


def predict_cup_csv(
    model_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    members: str,
    team: str,
    date: str,
) -> Path:
    """Predict every input row and write the four-comment ML-CUP25 CSV."""
    model_path = Path(model_path).expanduser().resolve()
    input_csv = Path(input_csv).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    if output_csv == model_path or output_csv == input_csv:
        raise ValueError("output CSV must differ from the model and input CSV")

    members = _header_value(members, "members")
    team = _header_value(team, "team")
    date = _submission_date(date)
    model, config, scaler_x, scaler_y, metadata = load_model_artifact(model_path)
    if metadata.get("cv_stop_reached") is False:
        warnings.warn("the saved model did not reach its CV stopping target; "
                      "it stopped at the epoch limit", RuntimeWarning, stacklevel=2)
    if config.get("task") != "cup" or config.get("n_inputs") != N_INPUTS:
        raise ValueError("model artifact is not a CUP model with 12 inputs")
    arch = config.get("arch", [])
    if not arch or arch[-1].get("units") != N_TARGETS:
        raise ValueError("CUP model artifact must have four outputs")
    scaled = bool(config.get("scale"))
    if scaled != (scaler_x is not None and scaler_y is not None):
        raise ValueError("model artifact scaling configuration is incomplete")
    if not scaled and (scaler_x is not None or scaler_y is not None):
        raise ValueError("unscaled CUP model must not contain scalers")

    ids, X = load_cup_blind(str(input_csv))
    X_fit = scaler_x.transform(X) if scaled else X
    predictions = model.predict(X_fit)
    if scaled:
        predictions = scaler_y.inverse_transform(predictions)
    if predictions.shape != (len(ids), N_TARGETS) or not np.isfinite(predictions).all():
        raise ValueError("model returned invalid CUP predictions")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_csv.parent,
            prefix=f".{output_csv.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(f"# {members}\n# {team}\n# ML-CUP25\n# {date}\n")
            writer = csv.writer(stream, lineterminator="\n")
            for identifier, values in zip(ids, predictions):
                writer.writerow([int(identifier), *(float(value) for value in values)])
        os.replace(temp_path, output_csv)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return output_csv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export CUP blind-test predictions as CSV.")
    parser.add_argument("--model", required=True, help="model.npz from the final CUP run")
    parser.add_argument("--input", required=True, help="blind CSV: id and 12 inputs")
    parser.add_argument("--output", required=True, help="output prediction CSV path")
    parser.add_argument("--members", required=True, help="comma-separated full names")
    parser.add_argument("--team", required=True, help="team name")
    parser.add_argument("--date", required=True, help="submission date as DD/MM/YYYY")
    args = parser.parse_args(argv)
    output = predict_cup_csv(args.model, args.input, args.output,
                             members=args.members, team=args.team, date=args.date)
    print(f"CUP predictions saved: {output}")


if __name__ == "__main__":
    main()
