"""CUP blind prediction produces the competition CSV without changing row IDs."""

import csv

import numpy as np
import pytest

from experiments.cup_predict import main, predict_cup_csv
from src.data.preprocessing import StandardScaler
from src.model_selection.build import build_model
from src.utils.model_artifact import save_model_artifact


def _saved_cup_model(tmp_path):
    config = {
        "task": "cup",
        "n_inputs": 12,
        "arch": [{"units": 4, "act": "identity"}],
        "optim": {"type": "sgd", "lr": 0.01},
        "loss": "mse",
        "scale": True,
        "seed": 19,
    }
    model = build_model(config)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    rng = np.random.default_rng(3)
    scaler_x.fit(rng.normal(size=(8, 12)))
    scaler_y.fit(rng.normal(size=(8, 4)) * 5 + 30)
    path = tmp_path / "model.npz"
    save_model_artifact(path, model, config, scaler_x, scaler_y)
    return path, model, scaler_x, scaler_y


def test_cli_exports_ids_and_original_unit_predictions(tmp_path, capsys):
    model_path, model, scaler_x, scaler_y = _saved_cup_model(tmp_path)
    X = np.arange(36, dtype=float).reshape(3, 12) / 10
    ids = [9, 2, 17]
    input_path = tmp_path / "blind.csv"
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# blind inputs\n")
        writer = csv.writer(stream, lineterminator="\n")
        for identifier, values in zip(ids, X):
            writer.writerow([identifier, *values])
    output_path = tmp_path / "submission.csv"

    main(["--model", str(model_path), "--input", str(input_path),
          "--output", str(output_path), "--members", "Ada Example, Bob Example",
          "--team", "Demo Team", "--date", "26/10/2026"])

    assert str(output_path) in capsys.readouterr().out
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[:4] == ["# Ada Example, Bob Example", "# Demo Team",
                         "# ML-CUP25", "# 26/10/2026"]
    assert len(lines) == 7
    assert all(not line.startswith("id,") for line in lines)
    assert b"\r\n" not in output_path.read_bytes()
    rows = list(csv.reader(lines[4:]))
    assert [int(row[0]) for row in rows] == ids
    assert all(len(row) == 5 for row in rows)
    actual = np.array([[float(value) for value in row[1:]] for row in rows])
    expected = scaler_y.inverse_transform(model.predict(scaler_x.transform(X)))
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_prediction_rejects_output_overwriting_an_input(tmp_path):
    model_path, _model, _scaler_x, _scaler_y = _saved_cup_model(tmp_path)
    input_path = tmp_path / "blind.csv"
    input_path.write_text("1," + ",".join(["0"] * 12) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must differ"):
        predict_cup_csv(model_path, input_path, input_path,
                        members="A B", team="Team", date="26/10/2026")
