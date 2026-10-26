"""Save and restore a fitted Network and its preprocessing as one NPZ file.

The archive contains only numeric arrays and JSON strings. Loading never needs
pickle, and the model is rebuilt from the saved config before weights are set.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..data.preprocessing import StandardScaler
from ..model_selection.build import build_model
from ..nn.network import Network


_SCHEMA_VERSION = 1


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _layer_sizes(config: dict) -> list[tuple[int, int]]:
    if not isinstance(config, dict):
        raise ValueError("model config must be a dictionary")
    n_in = _positive_int(config.get("n_inputs"), "config.n_inputs")
    arch = config.get("arch")
    if not isinstance(arch, list) or not arch:
        raise ValueError("config.arch must be a nonempty list")
    sizes = []
    for index, spec in enumerate(arch):
        if not isinstance(spec, dict) or not isinstance(spec.get("act"), str):
            raise ValueError(f"config.arch[{index}] needs an activation")
        n_out = _positive_int(spec.get("units"), f"config.arch[{index}].units")
        sizes.append((n_in, n_out))
        n_in = n_out
    return sizes


def _numeric_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _scaler_arrays(
    scaler: StandardScaler | None, n_features: int, name: str
) -> dict[str, np.ndarray]:
    if scaler is None:
        return {}
    if not isinstance(scaler, StandardScaler):
        raise ValueError(f"{name} must be a StandardScaler or None")
    mean = _numeric_array(scaler.mean, (n_features,), f"{name}.mean")
    scale = _numeric_array(scaler.scale, (n_features,), f"{name}.scale")
    if np.any(scale <= 0):
        raise ValueError(f"{name}.scale must be strictly positive")
    return {f"{name}_mean": mean, f"{name}_scale": scale}


def _json_string(value: dict, name: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    return np.array(encoded)


def save_model_artifact(
    path: str | Path,
    model: Network,
    config: dict,
    scaler_x: StandardScaler | None,
    scaler_y: StandardScaler | None,
    metadata: dict | None = None,
) -> Path:
    """Atomically save a fitted model and preprocessing; return the exact path."""
    if not isinstance(model, Network):
        raise ValueError("model must be a Network")
    sizes = _layer_sizes(config)
    if len(model.layers) != len(sizes):
        raise ValueError("model layer count does not match config.arch")

    payload: dict[str, np.ndarray] = {
        "schema_version": np.array(_SCHEMA_VERSION, dtype=np.int64),
        "config_json": _json_string(config, "config"),
        "metadata_json": _json_string(metadata if metadata is not None else {}, "metadata"),
    }
    for index, (layer, (n_in, n_out)) in enumerate(zip(model.layers, sizes)):
        if (layer.n_in, layer.n_out) != (n_in, n_out):
            raise ValueError(f"model layer {index} does not match config.arch")
        if layer.activation.__class__.__name__.lower() != config["arch"][index]["act"].lower():
            raise ValueError(f"model layer {index} activation does not match config.arch")
        payload[f"layer_{index}_W"] = _numeric_array(
            layer.W, (n_in, n_out), f"layer {index} weights"
        )
        payload[f"layer_{index}_b"] = _numeric_array(
            layer.b, (1, n_out), f"layer {index} bias"
        )

    payload.update(_scaler_arrays(scaler_x, sizes[0][0], "scaler_x"))
    payload.update(_scaler_arrays(scaler_y, sizes[-1][1], "scaler_y"))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.",
            suffix=".tmp", delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            np.savez_compressed(temporary, **payload)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return output_path


def _load_json(archive: np.lib.npyio.NpzFile, name: str) -> dict:
    value = archive[name]
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a JSON string")
    try:
        decoded = json.loads(value.item())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} contains invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return decoded


def _load_scaler(
    archive: np.lib.npyio.NpzFile, name: str, n_features: int
) -> StandardScaler | None:
    mean_key, scale_key = f"{name}_mean", f"{name}_scale"
    if (mean_key in archive.files) != (scale_key in archive.files):
        raise ValueError(f"{name} has incomplete statistics")
    if mean_key not in archive.files:
        return None
    mean = _numeric_array(archive[mean_key], (n_features,), mean_key)
    scale = _numeric_array(archive[scale_key], (n_features,), scale_key)
    if np.any(scale <= 0):
        raise ValueError(f"{scale_key} must be strictly positive")
    scaler = StandardScaler()
    scaler.mean = mean.astype(float, copy=True)
    scaler.scale = scale.astype(float, copy=True)
    return scaler


def load_model_artifact(
    path: str | Path,
) -> tuple[Network, dict, StandardScaler | None, StandardScaler | None, dict]:
    """Load and validate an archive without allowing pickled Python objects."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"schema_version", "config_json", "metadata_json"}
        if not required.issubset(archive.files):
            raise ValueError("model artifact is missing required fields")
        version = archive["schema_version"]
        if version.shape != () or version.dtype.kind not in "iu" or version.item() != _SCHEMA_VERSION:
            raise ValueError("unsupported model artifact schema version")
        config = _load_json(archive, "config_json")
        metadata = _load_json(archive, "metadata_json")
        sizes = _layer_sizes(config)

        expected = set(required)
        parameters = []
        for index, (n_in, n_out) in enumerate(sizes):
            weight_key, bias_key = f"layer_{index}_W", f"layer_{index}_b"
            expected.update((weight_key, bias_key))
            if weight_key not in archive.files or bias_key not in archive.files:
                raise ValueError(f"model artifact is missing layer {index} parameters")
            weights = _numeric_array(archive[weight_key], (n_in, n_out), weight_key)
            bias = _numeric_array(archive[bias_key], (1, n_out), bias_key)
            parameters.append((weights.copy(), bias.copy()))

        scaler_x = _load_scaler(archive, "scaler_x", sizes[0][0])
        scaler_y = _load_scaler(archive, "scaler_y", sizes[-1][1])
        if scaler_x is not None:
            expected.update(("scaler_x_mean", "scaler_x_scale"))
        if scaler_y is not None:
            expected.update(("scaler_y_mean", "scaler_y_scale"))
        if set(archive.files) != expected:
            raise ValueError("model artifact contains unexpected fields")

    try:
        model = build_model(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("model artifact has an invalid model config") from exc
    for layer, (weights, bias) in zip(model.layers, parameters):
        layer.W[...] = weights
        layer.b[...] = bias
    return model, config, scaler_x, scaler_y, metadata
