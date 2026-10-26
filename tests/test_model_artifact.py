"""Round-trip and validation tests for saved model artifacts."""

import numpy as np
import pytest

from src.data.preprocessing import StandardScaler
from src.model_selection.build import build_model
from src.utils.model_artifact import load_model_artifact, save_model_artifact


def _config():
    return {
        "n_inputs": 3,
        "arch": [
            {"units": 5, "act": "tanh"},
            {"units": 2, "act": "identity"},
        ],
        "init": "glorot",
        "loss": "mse",
        "optim": {"type": "sgd", "lr": 0.02},
        "seed": 42,
    }


def _trained_model():
    config = _config()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 3))
    y = x @ np.array([[1.0, 0.5], [-0.25, 0.4], [0.1, -0.8]])
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)
    model = build_model(config)
    model.fit(x_scaled, y_scaled, epochs=3, batch_size=4)
    return config, model, scaler_x, scaler_y, x


def test_round_trip_preserves_predictions_and_scalers(tmp_path):
    config, model, scaler_x, scaler_y, x = _trained_model()
    expected = scaler_y.inverse_transform(model.predict(scaler_x.transform(x)))
    path = tmp_path / "nested" / "model.npz"
    metadata = {"fit_seed": 123, "epochs_run": model.epochs_run}

    assert save_model_artifact(path, model, config, scaler_x, scaler_y, metadata) == path
    loaded, loaded_config, loaded_x, loaded_y, loaded_metadata = load_model_artifact(path)

    assert loaded_config == config
    assert loaded_metadata == metadata
    np.testing.assert_array_equal(loaded_x.mean, scaler_x.mean)
    np.testing.assert_array_equal(loaded_x.scale, scaler_x.scale)
    np.testing.assert_array_equal(loaded_y.mean, scaler_y.mean)
    np.testing.assert_array_equal(loaded_y.scale, scaler_y.scale)
    np.testing.assert_allclose(
        loaded_y.inverse_transform(loaded.predict(loaded_x.transform(x))),
        expected,
        rtol=0,
        atol=0,
    )
    with np.load(path, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == 1
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


def test_round_trip_without_scalers_uses_exact_path(tmp_path):
    config = _config()
    model = build_model(config)
    path = tmp_path / "model_without_suffix"
    save_model_artifact(path, model, config, None, None)

    assert path.exists()
    loaded, loaded_config, scaler_x, scaler_y, metadata = load_model_artifact(path)
    assert loaded_config == config
    assert metadata == {}
    assert scaler_x is None and scaler_y is None
    np.testing.assert_array_equal(loaded.predict(np.ones((2, 3))), model.predict(np.ones((2, 3))))


def test_save_rejects_invalid_weights_and_scaler(tmp_path):
    config, model, scaler_x, scaler_y, _ = _trained_model()
    model.layers[0].W[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        save_model_artifact(tmp_path / "bad_weights.npz", model, config, scaler_x, scaler_y)

    model.layers[0].W[0, 0] = 0.0
    scaler_y.scale[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        save_model_artifact(tmp_path / "bad_scaler.npz", model, config, scaler_x, scaler_y)


def _rewrite(path, changes):
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload.update(changes)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def test_load_rejects_unknown_schema_and_parameter_shape(tmp_path):
    config, model, scaler_x, scaler_y, _ = _trained_model()
    path = tmp_path / "model.npz"
    save_model_artifact(path, model, config, scaler_x, scaler_y)

    _rewrite(path, {"schema_version": np.array(99)})
    with pytest.raises(ValueError, match="schema version"):
        load_model_artifact(path)

    save_model_artifact(path, model, config, scaler_x, scaler_y)
    _rewrite(path, {"layer_0_W": np.ones((2, 5))})
    with pytest.raises(ValueError, match="expected"):
        load_model_artifact(path)


def test_load_rejects_object_array_without_pickle(tmp_path):
    config, model, _, _, _ = _trained_model()
    path = tmp_path / "model.npz"
    save_model_artifact(path, model, config, None, None)
    _rewrite(path, {"layer_0_W": np.array([object()], dtype=object)})
    with pytest.raises(ValueError, match="allow_pickle=False"):
        load_model_artifact(path)
