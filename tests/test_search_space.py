"""Contract for expanding compact search ranges into model configurations."""

import json
from collections import Counter
from pathlib import Path

import pytest

from runner.engine import run_experiment
from runner.search_space import iter_search_configs, iter_values
from src.model_selection.build import build_model


def _fixed():
    return {
        "n_inputs": 17,
        "loss": "mse",
        "epochs": 12,
        "batch_size": None,
        "init": "glorot",
        "reg": None,
    }


def _search():
    return {
        "hidden_layers": {"scale": "int", "start": 1, "stop": 2},
        "hidden_units": [
            {"scale": "int", "start": 1, "stop": 8},
            {"scale": "int", "start": 1, "stop": 4},
        ],
        "hidden_activation": "tanh",
        "output": {"units": 1, "activation": "sigmoid"},
        "optimizer": {
            "type": "sgd",
            "lr": {"scale": "log", "start": 0.1, "stop": 0.00001, "factor": 10},
            "momentum": 0.9,
        },
    }


def test_sample_config_null_options_are_disabled_and_can_be_enabled():
    path = (Path(__file__).resolve().parents[1] / "configs" / "samples"
            / "monk_mod_selection.json")
    sample = json.loads(path.read_text(encoding="utf-8"))
    payload = sample["config"]
    fixed = {**payload["fixed"], "n_inputs": 17}
    configs = list(iter_search_configs(fixed, payload["search"]))

    assert len(configs) == 12
    assert all(config["reg"] is None for config in configs)
    assert all(set(config["optim"]) == {"type", "lr"} for config in configs)
    model = build_model(configs[0])
    assert model.regularizer is None
    assert model.optimizer.momentum == 0.0
    assert model.optimizer.nesterov is False

    enabled_fixed = {**fixed, "init": "uniform", "init_kwargs": {"scale": 0.3},
                     "reg": {"type": "l2", "lam": 0.01}}
    enabled_search = {**payload["search"],
                      "optimizer": {**payload["search"]["optimizer"],
                                    "momentum": 0.9, "nesterov": True}}
    enabled = next(iter_search_configs(enabled_fixed, enabled_search))
    enabled_model = build_model(enabled)
    assert enabled_model.layers[0].initializer.scale == 0.3
    assert enabled_model.regularizer.lam == 0.01
    assert enabled_model.optimizer.momentum == 0.9
    assert enabled_model.optimizer.nesterov is True

    for kind, setting, default in (("adagrad", "epsilon", 1e-8),
                                   ("quickprop", "mu", 1.75)):
        alternative = {**payload["search"],
                       "optimizer": {**payload["search"]["optimizer"], "type": kind}}
        config = next(iter_search_configs(fixed, alternative))
        assert getattr(build_model(config).optimizer, setting) == default


def test_iter_values_includes_integer_bounds_and_descending_log_bounds():
    assert list(iter_values({"scale": "int", "start": 1, "stop": 3})) == [1, 2, 3]
    assert list(iter_values({"scale": "int", "start": 3, "stop": 1})) == [3, 2, 1]
    assert list(iter_values({"scale": "log", "start": 0.1,
                             "stop": 0.00001, "factor": 10})) == pytest.approx(
                                 [0.1, 0.01, 0.001, 0.0001, 0.00001])
    assert list(iter_values({"scale": "log", "start": 0.001,
                             "stop": 0.1})) == pytest.approx([0.001, 0.01, 0.1])
    assert list(iter_values({"scale": "linear", "start": 0,
                             "stop": 0.9, "num": 4})) == pytest.approx([0, 0.3, 0.6, 0.9])
    assert list(iter_values({"scale": "linear", "start": 0.9,
                             "stop": 0, "num": 4})) == pytest.approx([0.9, 0.6, 0.3, 0])
    assert list(iter_values({"scale": "linear", "start": 0,
                             "stop": 0.99, "step": 0.33})) == pytest.approx(
                                 [0, 0.33, 0.66, 0.99])
    assert list(iter_values({"scale": "linear", "start": 0.99,
                             "stop": 0, "step": 0.33})) == pytest.approx(
                                 [0.99, 0.66, 0.33, 0])


def test_compact_search_expands_to_200_unique_buildable_configurations():
    fixed = _fixed()
    search = _search()

    configs = list(iter_search_configs(fixed, search))

    assert len(configs) == 200
    assert len({json.dumps(config, sort_keys=True) for config in configs}) == 200
    assert Counter(len(config["arch"]) - 1 for config in configs) == {1: 40, 2: 160}
    assert len({tuple(layer["units"] for layer in config["arch"][:-1])
                for config in configs}) == 40
    assert sorted(Counter(config["optim"]["lr"] for config in configs).values()) == [40] * 5

    for config in configs:
        assert all(config[key] == value for key, value in fixed.items())
        hidden = config["arch"][:-1]
        assert 1 <= hidden[0]["units"] <= 8
        if len(hidden) == 2:
            assert 1 <= hidden[1]["units"] <= 4
        assert all(layer["act"] == "tanh" for layer in hidden)
        assert config["arch"][-1]["units"] == 1
        assert config["arch"][-1]["act"] == "sigmoid"
        assert config["optim"]["type"] == "sgd"
        assert config["optim"]["momentum"] == 0.9
        assert isinstance(config["optim"]["lr"], float)

    for config in (configs[0], configs[40], configs[-1]):
        assert len(build_model(config).layers) == len(config["arch"])


def test_compact_search_expands_regularization_strengths():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 2, "stop": 2}]
    search["optimizer"]["lr"] = {"scale": "log", "start": 0.1, "stop": 0.1}
    search["reg"] = {"type": "l2", "lam": {
        "scale": "log", "start": 0.1, "stop": 0.0001,
    }}

    configs = list(iter_search_configs(_fixed(), search))

    assert [config["reg"]["lam"] for config in configs] == pytest.approx(
        [0.1, 0.01, 0.001, 0.0001])
    assert all(build_model(config).regularizer.lam == config["reg"]["lam"]
               for config in configs)


def test_fine_grid_compact_search_uses_nesterov_and_regularization_options():
    path = (Path(__file__).resolve().parents[1] / "configs" / "monk1_sgd"
            / "monk1_grid_fine.json")
    payload = json.loads(path.read_text(encoding="utf-8"))["config"]
    configs = list(iter_search_configs({**payload["fixed"], "n_inputs": 17},
                                       payload["search"]))

    assert len(configs) == 5
    assert len({json.dumps(config, sort_keys=True) for config in configs}) == 5
    for config in configs:
        assert config["arch"] == [{"units": 4, "act": "tanh"},
                                  {"units": 1, "act": "sigmoid"}]
        assert config["optim"]["lr"] == 0.1
        model = build_model(config)
        assert model.optimizer.momentum == config["optim"].get("momentum", 0)
        assert model.optimizer.nesterov == config["optim"].get("nesterov", False)
        assert (model.regularizer is None) == (config["reg"] is None)
    assert {(config["optim"]["momentum"], config["optim"]["nesterov"])
            for config in configs} == {(0.9, True)}
    assert {None if config["reg"] is None else config["reg"]["lam"]
            for config in configs} == {None, 0.00001, 0.0001, 0.001, 0.01}


def test_compact_search_interpolates_momentum_with_nesterov():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 4, "stop": 4}]
    search["optimizer"]["lr"] = {"scale": "log", "start": 0.1, "stop": 0.1}
    search["optimizer"]["momentum"] = {
        "scale": "linear", "start": 0, "stop": 0.99, "step": 0.33}
    search["optimizer"]["nesterov"] = True

    configs = list(iter_search_configs(_fixed(), search))

    assert len(configs) == 4
    assert {(config["optim"].get("momentum", 0),
             config["optim"].get("nesterov", False)) for config in configs} == {
                 (0, False), (0.33, True), (0.66, True), (0.99, True)}


@pytest.mark.parametrize("momenta", [[], [0, 0], [1], [-0.1], [True]])
def test_compact_search_rejects_invalid_momentum_lists(momenta):
    search = _search()
    search["optimizer"]["momentum"] = momenta
    with pytest.raises(ValueError, match="momentum"):
        list(iter_search_configs(_fixed(), search))


def test_compact_search_rejects_non_boolean_include_none():
    search = _search()
    search["reg"] = {"type": "l2", "lam": {"scale": "log", "start": 0.01,
                                             "stop": 0.001}, "include_none": "yes"}
    with pytest.raises(ValueError, match="include_none"):
        list(iter_search_configs(_fixed(), search))


def test_compact_search_rejects_regularization_inside_optimizer():
    search = _search()
    search["optimizer"]["reg"] = {"type": "l2", "lam": 0.01}
    with pytest.raises(ValueError, match="reg belongs in search"):
        list(iter_search_configs(_fixed(), search))


def test_compact_search_rejects_conflicting_fixed_regularization():
    search = _search()
    search["reg"] = {"type": "l2", "lam": {
        "scale": "log", "start": 0.1, "stop": 0.01,
    }}
    fixed = {**_fixed(), "reg": {"type": "l1", "lam": 0.01}}
    with pytest.raises(ValueError, match="both fixed and search"):
        list(iter_search_configs(fixed, search))


@pytest.mark.parametrize("spec", [
    {"scale": "int", "start": "1", "stop": 3},
    {"scale": "int", "start": 1.5, "stop": 3},
    {"scale": "log", "start": 0, "stop": 0.1, "factor": 10},
    {"scale": "log", "start": 0.1, "stop": -0.01, "factor": 10},
    {"scale": "log", "start": 0.1, "stop": 0.001, "factor": 1},
    {"scale": "linear", "start": 0, "stop": 0.9, "num": 0},
    {"scale": "linear", "start": 0, "stop": 0.9, "num": True},
    {"scale": "linear", "start": 0, "stop": 0.9, "num": 1},
    {"scale": "linear", "start": 0.9, "stop": 0.9, "num": 4},
    {"scale": "linear", "start": 0, "stop": 0.9, "step": 0},
    {"scale": "linear", "start": 0, "stop": 0.9, "step": 0.4},
    {"scale": "linear", "start": 0, "stop": 0.9, "step": 0.3, "num": 4},
    {"scale": "unknown", "start": 1, "stop": 2},
])
def test_iter_values_rejects_invalid_ranges(spec):
    with pytest.raises(ValueError):
        list(iter_values(spec))


@pytest.mark.parametrize("layer_spec, unit_specs", [
    ({"scale": "int", "start": 0, "stop": 1},
     [{"scale": "int", "start": 1, "stop": 8}]),
    ({"scale": "int", "start": 1, "stop": 1},
     [{"scale": "int", "start": 0, "stop": 8}]),
    ({"scale": "int", "start": 2, "stop": 3},
     [{"scale": "int", "start": 1, "stop": 8},
      {"scale": "int", "start": 1, "stop": 4}]),
])
def test_compact_search_rejects_invalid_architectures(layer_spec, unit_specs):
    search = _search()
    search["hidden_layers"] = layer_spec
    search["hidden_units"] = unit_specs
    with pytest.raises(ValueError):
        list(iter_search_configs(_fixed(), search))


def test_log_range_can_choose_integer_hidden_layer_counts():
    search = _search()
    search["hidden_layers"] = {"scale": "log", "start": 1, "stop": 4, "factor": 2}
    search["hidden_units"] = [
        {"scale": "int", "start": 2, "stop": 2} for _ in range(4)
    ]
    search["optimizer"]["lr"] = {"scale": "log", "start": 0.1, "stop": 0.1}

    configs = list(iter_search_configs(_fixed(), search))

    assert [len(config["arch"]) - 1 for config in configs] == [1, 2, 4]
    assert all(type(layer["units"]) is int
               for config in configs for layer in config["arch"])


def test_log_range_can_choose_integer_hidden_unit_counts():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [
        {"scale": "log", "start": 1, "stop": 8, "factor": 2}
    ]
    search["optimizer"]["lr"] = {"scale": "log", "start": 0.1, "stop": 0.1}

    configs = list(iter_search_configs(_fixed(), search))

    assert [config["arch"][0]["units"] for config in configs] == [1, 2, 4, 8]
    assert all(type(config["arch"][0]["units"]) is int for config in configs)
    assert all(len(build_model(config).layers) == 2 for config in configs)


@pytest.mark.parametrize("field", ["hidden_layers", "hidden_units"])
def test_log_range_rejects_fractional_architecture_values(field):
    search = _search()
    fractional = {"scale": "log", "start": 1, "stop": 3, "factor": 1.5}
    if field == "hidden_layers":
        search["hidden_layers"] = fractional
        search["hidden_units"] = [
            {"scale": "int", "start": 2, "stop": 2} for _ in range(3)
        ]
    else:
        search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
        search["hidden_units"] = [fractional]
    with pytest.raises(ValueError):
        list(iter_search_configs(_fixed(), search))


def _tiny_select_payload():
    fixed = _fixed()
    fixed["epochs"] = 2
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 2, "stop": 2}]
    search["optimizer"]["lr"] = {
        "scale": "log", "start": 0.1, "stop": 0.01, "factor": 10,
    }
    return {
        "seed": 0, "n_core": 1, "val_fraction": 0.25,
        "fixed": fixed, "search": search,
    }


def test_run_experiment_accepts_compact_search_and_saves_its_source(tmp_path):
    payload = _tiny_select_payload()

    run_dir = run_experiment("monk1", "select", payload,
                             tag="compact-test", results_root=tmp_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    assert summary["validation_mode"] == "holdout"
    assert summary["n_configs"] == len(summary["ranking"]) == 2
    assert {row["config"]["optim"]["lr"] for row in summary["ranking"]} == {0.01, 0.1}
    for row in summary["ranking"]:
        config = row["config"]
        assert config["n_inputs"] == 17
        assert config["epochs"] == 2
        assert config["arch"] == [
            {"units": 2, "act": "tanh"},
            {"units": 1, "act": "sigmoid"},
        ]
        assert config["optim"]["type"] == "sgd"
        assert config["optim"]["momentum"] == 0.9
        assert {"train_error", "val_error", "val_mean"} <= row.keys()
    assert saved["search"] == payload["search"]
    assert saved["fixed"] == payload["fixed"]
    assert "grid" not in saved


@pytest.mark.parametrize("include_grid, include_search", [(True, True), (False, False)])
def test_select_requires_exactly_one_search_input(tmp_path, include_grid, include_search):
    payload = _tiny_select_payload()
    payload.pop("search")
    if include_grid:
        payload["grid"] = {}
    if include_search:
        payload["search"] = _search()

    with pytest.raises(ValueError, match="exactly one of grid or search"):
        run_experiment("monk1", "select", payload, results_root=tmp_path)
    assert list(tmp_path.iterdir()) == []
