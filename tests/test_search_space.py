"""Contract for expanding compact search ranges into model configurations"""

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


def test_null_options_are_disabled_and_can_be_enabled():
    fixed = {**_fixed(), "init_kwargs": None, "scale": None,
             "reg": {"type": "l2", "lam": None}}
    search = _search()
    search["hidden_units"] = [
        {"scale": "int", "start": 1, "stop": 2},
        {"scale": "int", "start": 1, "stop": 2},
    ]
    search["optimizer"].update({
        "lr": {"scale": "log", "start": 0.1, "stop": 0.01, "factor": 10},
        "momentum": None,
    })
    configs = list(iter_search_configs(fixed, search))

    assert len(configs) == 12
    assert all(config["reg"] is None for config in configs)
    assert all(set(config["optim"]) == {"type", "lr"} for config in configs)
    model = build_model(configs[0])
    assert model.regularizer is None
    assert model.optimizer.momentum == 0.0
    assert model.optimizer.nesterov is False

    enabled_fixed = {**fixed, "init": "uniform", "init_kwargs": {"scale": 0.3},
                     "reg": {"type": "l2", "lam": 0.01}}
    enabled_search = {**search,
                      "optimizer": {**search["optimizer"],
                                    "momentum": {"scale": "linear", "start": 0.9,
                                                 "stop": 0.9, "num": 1,
                                                 "nesterov": True}}}
    enabled = next(iter_search_configs(enabled_fixed, enabled_search))
    enabled_model = build_model(enabled)
    assert enabled_model.layers[0].initializer.scale == 0.3
    assert enabled_model.regularizer.lam == 0.01
    assert enabled_model.optimizer.momentum == 0.9
    assert enabled_model.optimizer.nesterov is True


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


def test_linear_range_can_append_an_unaligned_stop():
    assert list(iter_values({"scale": "linear", "start": 0.01,
                             "stop": 0.9, "step": 0.2,
                             "include_stop": True})) == pytest.approx(
                                 [0.01, 0.21, 0.41, 0.61, 0.81, 0.9])


def test_top3_compact_search_preserves_the_requested_grid():
    path = (Path(__file__).resolve().parents[1] / "configs" / "monk1"
            / "monk1_grid_coarse_top3.json")
    payload = json.loads(path.read_text(encoding="utf-8"))["config"]
    configs = list(iter_search_configs({**payload["fixed"], "n_inputs": 17},
                                       payload["search"]))

    assert len(configs) == 480
    assert {tuple(layer["units"] for layer in config["arch"]) for config in configs} == {
        (8, 1), (8, 2, 1), (8, 4, 1)}
    assert {config["optim"]["lr"] for config in configs} == {
        0.001, 0.01, 0.1, 1.0}
    assert {config["optim"].get("momentum", 0) for config in configs} == {
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    assert {None if config["reg"] is None else config["reg"]["lam"]
            for config in configs} == {None, 0.0001, 0.001, 0.01}


def test_cup_coarse2_search_keeps_exact_architectures_and_rates():
    path = (Path(__file__).resolve().parents[1] / "configs" / "cup"
            / "cup_grid_coarse2.json")
    payload = json.loads(path.read_text(encoding="utf-8"))["config"]
    configs = list(iter_search_configs({**payload["fixed"], "n_inputs": 10},
                                       payload["search"]))

    assert len(configs) == 20
    assert {tuple(layer["units"] for layer in config["arch"][:-1])
            for config in configs} == {
                (128, 64), (128, 128), (64, 64, 64),
                (128, 64, 32), (128, 128, 128),
            }
    assert {config["optim"]["lr"] for config in configs} == {
        0.0001, 0.001, 0.01, 0.1,
    }
    assert all(config["optim"]["type"] == "sgd" for config in configs)


@pytest.mark.parametrize("architectures", [
    [], [[]], [[0]], [[True]], [[64.0]], [[64], [64]],
])
def test_explicit_architectures_reject_invalid_or_duplicate_choices(architectures):
    search = _search()
    del search["hidden_layers"]
    del search["hidden_units"]
    search["architectures"] = architectures

    with pytest.raises(ValueError, match="architectures"):
        list(iter_search_configs(_fixed(), search))


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
    path = (Path(__file__).resolve().parents[1] / "configs" / "monk1"
            / "monk1_grid_fine.json")
    payload = json.loads(path.read_text(encoding="utf-8"))["config"]
    configs = list(iter_search_configs({**payload["fixed"], "n_inputs": 17},
                                       payload["search"]))

    assert len(configs) == 48
    assert len({json.dumps(config, sort_keys=True) for config in configs}) == 48
    for config in configs:
        assert config["arch"] == [{"units": 4, "act": "tanh"},
                                  {"units": 1, "act": "sigmoid"}]
        model = build_model(config)
        assert model.optimizer.momentum == config["optim"].get("momentum", 0)
        assert model.optimizer.nesterov == config["optim"].get("nesterov", False)
        assert (model.regularizer is None) == (config["reg"] is None)
    assert {config["optim"]["lr"] for config in configs} == {0.005, 0.01, 0.05, 0.08}
    assert {(config["optim"].get("momentum", 0),
             config["optim"].get("nesterov", False)) for config in configs} == {
                 (0.85, False), (0.9, False), (0.95, False)}
    assert {None if config["reg"] is None else config["reg"]["lam"]
            for config in configs} == {None, 0.0001, 0.001, 0.01}


def test_compact_search_interpolates_momentum_with_nesterov():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 4, "stop": 4}]
    search["optimizer"]["lr"] = {"scale": "log", "start": 0.1, "stop": 0.1}
    search["optimizer"]["momentum"] = {
        "scale": "linear", "start": 0, "stop": 0.99, "step": 0.33,
        "nesterov": True}

    configs = list(iter_search_configs(_fixed(), search))

    assert len(configs) == 4
    assert {(config["optim"].get("momentum", 0),
             config["optim"].get("nesterov", False)) for config in configs} == {
                 (0, False), (0.33, True), (0.66, True), (0.99, True)}


def test_nested_nesterov_is_ignored_when_include_zero_disables_momentum():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 4, "stop": 4}]
    search["optimizer"]["lr"] = [0.1]
    search["optimizer"]["momentum"] = {
        "scale": "linear", "start": 0.7, "stop": 0.9, "step": 0.1,
        "nesterov": True, "include_zero": True,
    }

    configs = list(iter_search_configs(_fixed(), search))

    assert len(configs) == 4
    zero, *enabled = configs
    assert zero["optim"] == {"type": "sgd", "lr": 0.1}
    assert build_model(zero).optimizer.momentum == 0
    assert build_model(zero).optimizer.nesterov is False
    assert [(config["optim"]["momentum"], config["optim"]["nesterov"])
            for config in enabled] == [(0.7, True), (0.8, True), (0.9, True)]


@pytest.mark.parametrize("nesterov", [False, True])
def test_nested_nesterov_scalar_generates_one_variant(nesterov):
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 4, "stop": 4}]
    search["optimizer"]["lr"] = [0.1]
    search["optimizer"]["momentum"] = {
        "values": [0.8], "nesterov": nesterov,
    }

    configs = list(iter_search_configs(_fixed(), search))

    assert len(configs) == 1
    assert configs[0]["optim"] == {
        "type": "sgd", "lr": 0.1, "momentum": 0.8, "nesterov": nesterov,
    }
    assert build_model(configs[0]).optimizer.nesterov is nesterov


def test_nested_nesterov_list_expands_only_positive_momenta():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 4, "stop": 4}]
    search["optimizer"]["lr"] = [0.1]
    search["optimizer"]["momentum"] = {
        "values": [0.7, 0.8], "nesterov": [False, True], "include_zero": True,
    }

    configs = list(iter_search_configs(_fixed(), search))

    assert len(configs) == len({json.dumps(config, sort_keys=True) for config in configs}) == 5
    assert [config["optim"] for config in configs] == [
        {"type": "sgd", "lr": 0.1},
        {"type": "sgd", "lr": 0.1, "momentum": 0.7, "nesterov": False},
        {"type": "sgd", "lr": 0.1, "momentum": 0.7, "nesterov": True},
        {"type": "sgd", "lr": 0.1, "momentum": 0.8, "nesterov": False},
        {"type": "sgd", "lr": 0.1, "momentum": 0.8, "nesterov": True},
    ]
    assert [build_model(config).optimizer.nesterov for config in configs] == [
        False, False, True, False, True,
    ]


def test_nested_nesterov_list_does_not_repeat_explicit_zero():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 4, "stop": 4}]
    search["optimizer"]["lr"] = [0.1]
    search["optimizer"]["momentum"] = {
        "values": [0, 0.8], "nesterov": [False, True], "include_zero": True,
    }

    configs = list(iter_search_configs(_fixed(), search))

    assert len(configs) == 3
    assert sum("momentum" not in config["optim"] for config in configs) == 1
    assert sum("nesterov" not in config["optim"] for config in configs) == 1


@pytest.mark.parametrize("nesterov", [
    None, 0, "true", [], [True, True], [False, False], [True, 0], ["false", True],
])
def test_nested_nesterov_rejects_invalid_option_lists(nesterov):
    search = _search()
    search["optimizer"]["momentum"] = {
        "values": [0.8], "nesterov": nesterov,
    }

    with pytest.raises(ValueError, match="nesterov"):
        list(iter_search_configs(_fixed(), search))


def test_nested_nesterov_rejects_invalid_values():
    search = _search()
    search["optimizer"]["momentum"] = {
        "scale": "linear", "start": 0.7, "stop": 0.9, "step": 0.1,
        "nesterov": "yes",
    }
    with pytest.raises(ValueError, match="nesterov must be a boolean"):
        list(iter_search_configs(_fixed(), search))


@pytest.mark.parametrize("old_value", [True, False, None])
def test_search_rejects_nesterov_outside_momentum(old_value):
    search = _search()
    search["optimizer"]["nesterov"] = old_value

    with pytest.raises(ValueError, match="nesterov must be set inside"):
        list(iter_search_configs(_fixed(), search))


def test_nested_nesterov_with_explicit_momentum_values():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 4, "stop": 4}]
    search["optimizer"]["lr"] = [0.1]
    search["optimizer"]["momentum"] = {
        "values": [0.6, 0.7, 0.8, 0.9],
        "nesterov": True, "include_zero": True,
    }

    configs = list(iter_search_configs(_fixed(), search))

    assert [(config["optim"].get("momentum", 0),
             config["optim"].get("nesterov", False)) for config in configs] == [
                 (0, False), (0.6, True), (0.7, True), (0.8, True), (0.9, True)]


@pytest.mark.parametrize("relative_path, expected_momenta", [
    ("monk1/monk1_grid_coarse_top1.json", {0, 0.6, 0.7, 0.8, 0.9}),
    ("monk2/monk2_grid_fine1.json", {0.7, 0.75, 0.8, 0.85, 0.9}),
    ("monk2/monk2_grid_coarse_top1.json", {0, 0.5, 0.7, 0.9}),
    ("monk2/monk2_grid_coarse2_full.json", {0, 0.5, 0.6, 0.7, 0.8, 0.9}),
])
def test_existing_nested_nesterov_searches(relative_path, expected_momenta):
    path = Path(__file__).resolve().parents[1] / "configs" / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))["config"]
    configs = list(iter_search_configs({**payload["fixed"], "n_inputs": 17},
                                       payload["search"]))

    assert {config["optim"].get("momentum", 0) for config in configs} == expected_momenta
    assert all(config["optim"].get("nesterov", False)
               == (config["optim"].get("momentum", 0) > 0)
               for config in configs)


def test_compact_search_accepts_explicit_learning_rates():
    search = _search()
    search["hidden_layers"] = {"scale": "int", "start": 1, "stop": 1}
    search["hidden_units"] = [{"scale": "int", "start": 4, "stop": 4}]
    search["optimizer"]["lr"] = [0.01, 0.1, 0.5, 0.8]

    configs = list(iter_search_configs(_fixed(), search))

    assert len(configs) == 4
    assert [config["optim"]["lr"] for config in configs] == [0.01, 0.1, 0.5, 0.8]


@pytest.mark.parametrize("rates", [[], [0.1, 0.1], [0], [-0.1], [True]])
def test_compact_search_rejects_invalid_learning_rate_lists(rates):
    search = _search()
    search["optimizer"]["lr"] = rates
    with pytest.raises(ValueError, match="learning rate"):
        list(iter_search_configs(_fixed(), search))


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
