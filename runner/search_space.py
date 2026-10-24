"""Interpret compact model-selection ranges as concrete model configurations."""

from __future__ import annotations

from decimal import Decimal
from itertools import product
from math import isfinite
from typing import Iterator


def _decimal(value, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{name} must be a number")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError(f"{name} must be finite")
    return number


def iter_values(spec: dict) -> Iterator[int | float]:
    """Yield inclusive integer, linear, or geometric range values."""
    if not isinstance(spec, dict):
        raise ValueError("range specification must be an object")
    scale = spec.get("scale")
    if scale == "int":
        if set(spec) != {"scale", "start", "stop"}:
            raise ValueError("int range requires only scale, start and stop")
        start, stop = spec["start"], spec["stop"]
        if (isinstance(start, bool) or isinstance(stop, bool)
                or not isinstance(start, int) or not isinstance(stop, int)):
            raise ValueError("int range endpoints must be integers")
        step = 1 if start <= stop else -1
        yield from range(start, stop + step, step)
    elif scale == "linear":
        if set(spec) not in ({"scale", "start", "stop", "num"},
                             {"scale", "start", "stop", "step"},
                             {"scale", "start", "stop", "step", "include_stop"}):
            raise ValueError("linear range requires scale, start, stop and either num or step")
        start = _decimal(spec["start"], "linear start")
        stop = _decimal(spec["stop"], "linear stop")
        if "num" in spec:
            num = spec["num"]
            if (isinstance(num, bool) or not isinstance(num, int) or num < 1
                    or (num == 1) != (start == stop)):
                raise ValueError("linear range needs distinct endpoints for num >= 2")
            step = (stop - start) / (num - 1) if num > 1 else Decimal(0)
        else:
            magnitude = _decimal(spec["step"], "linear step")
            if magnitude <= 0:
                raise ValueError("linear step must be positive")
            distance = abs(stop - start)
            include_stop = spec.get("include_stop", False)
            if not isinstance(include_stop, bool):
                raise ValueError("linear include_stop must be a boolean")
            remainder = distance % magnitude
            if remainder and not include_stop:
                raise ValueError("linear step must reach stop exactly")
            num = int(distance // magnitude) + 1
            step = magnitude if stop >= start else -magnitude
        for index in range(num):
            value = (stop if index == num - 1 and ("num" in spec or not remainder)
                     else start + index * step)
            try:
                result = float(value)
            except OverflowError as exc:
                raise ValueError("linear range contains a value outside floating-point limits") from exc
            if not isfinite(result):
                raise ValueError("linear range contains a value outside floating-point limits")
            yield int(value) if value == value.to_integral_value() else result
        if "step" in spec and remainder and include_stop:
            result = float(stop)
            if not isfinite(result):
                raise ValueError("linear range contains a value outside floating-point limits")
            yield int(stop) if stop == stop.to_integral_value() else result
    elif scale == "log":
        if not {"scale", "start", "stop"} <= spec.keys() or set(spec) - {
                "scale", "start", "stop", "factor"}:
            raise ValueError("log range requires scale, start, stop and optional factor")
        start = _decimal(spec["start"], "log start")
        stop = _decimal(spec["stop"], "log stop")
        factor = _decimal(spec.get("factor", 10), "log factor")
        if start <= 0 or stop <= 0:
            raise ValueError("log range endpoints must be positive")
        if factor <= 1:
            raise ValueError("log factor must be greater than 1")
        ascending = start <= stop
        value = start
        while ((ascending and value <= stop) or
               (not ascending and value >= stop)):
            try:
                result = float(value)
            except OverflowError as exc:
                raise ValueError("log range contains a value outside floating-point limits") from exc
            if result == 0 or not isfinite(result):
                raise ValueError("log range contains a value outside floating-point limits")
            yield int(value) if value == value.to_integral_value() else result
            value = value * factor if ascending else value / factor
    else:
        raise ValueError(f"unknown range scale: {scale!r}")


def iter_search_configs(fixed: dict, search: dict) -> Iterator[dict]:
    """Yield complete build_model configurations directly from a compact search."""
    if not isinstance(fixed, dict) or not isinstance(search, dict):
        raise ValueError("fixed and search must be objects")
    expected = {"hidden_layers", "hidden_units", "hidden_activation", "output",
                "optimizer"}
    if set(search) - {"reg"} != expected:
        raise ValueError(f"search must contain {sorted(expected)} and optional reg")
    if "arch" in fixed or "optim" in fixed:
        raise ValueError("arch and optim belong in search, not fixed")
    if "reg" in search and fixed.get("reg") is not None:
        raise ValueError("reg cannot be set in both fixed and search")

    depths = list(iter_values(search["hidden_layers"]))
    if any(not isinstance(depth, int) or depth < 1 for depth in depths):
        raise ValueError("hidden_layers must contain positive integers")
    unit_specs = search["hidden_units"]
    if not isinstance(unit_specs, list) or len(unit_specs) != max(depths):
        raise ValueError("hidden_units needs one range for each possible hidden layer")
    widths = [list(iter_values(spec)) for spec in unit_specs]
    if any(not all(isinstance(unit, int) and unit > 0 for unit in axis)
           for axis in widths):
        raise ValueError("hidden_units ranges must contain positive integers")

    hidden_act = search["hidden_activation"]
    output = search["output"]
    optimizer = search["optimizer"]
    if not isinstance(hidden_act, str) or not hidden_act:
        raise ValueError("hidden_activation must be a nonempty string")
    if not isinstance(output, dict) or set(output) != {"units", "activation"}:
        raise ValueError("output requires units and activation")
    if (isinstance(output["units"], bool) or not isinstance(output["units"], int)
            or output["units"] < 1 or not isinstance(output["activation"], str)
            or not output["activation"]):
        raise ValueError("output needs positive integer units and an activation")
    if (not isinstance(optimizer, dict) or "lr" not in optimizer
            or not isinstance(optimizer.get("type"), str) or not optimizer["type"]):
        raise ValueError("optimizer needs type and a learning-rate range or list")
    if "reg" in optimizer:
        raise ValueError("reg belongs in search, not search.optimizer")
    rate_spec = optimizer["lr"]
    if isinstance(rate_spec, list):
        if not rate_spec:
            raise ValueError("learning rates require a nonempty list")
        rates = []
        for value in rate_spec:
            number = _decimal(value, "learning rate")
            try:
                rate = float(number)
            except OverflowError as exc:
                raise ValueError("learning rate is outside floating-point limits") from exc
            if not isfinite(rate) or rate <= 0:
                raise ValueError("learning rates must be positive and finite")
            rates.append(rate)
        if len(set(rates)) != len(rates):
            raise ValueError("learning rates must be unique")
    else:
        rates = list(iter_values(rate_spec))
        if any(rate <= 0 for rate in rates):
            raise ValueError("learning rates must be positive")
    # Null optional optimizer settings use the selected optimizer's defaults.
    momentum_spec = optimizer.get("momentum")
    if isinstance(momentum_spec, (list, dict)):
        if optimizer["type"] != "sgd":
            raise ValueError("momentum values require SGD")
        nesterov = optimizer.get("nesterov")
        if nesterov is not None and not isinstance(nesterov, bool):
            raise ValueError("nesterov must be a boolean")
        if isinstance(momentum_spec, dict):
            include_zero = momentum_spec.get("include_zero", False)
            if not isinstance(include_zero, bool):
                raise ValueError("momentum include_zero must be a boolean")
            values = list(iter_values({key: value for key, value in momentum_spec.items()
                                       if key != "include_zero"}))
            if include_zero:
                values.insert(0, 0)
        else:
            values = momentum_spec
        if not values:
            raise ValueError("momentum values require a nonempty range or list")
        momenta = [float(_decimal(value, "momentum")) for value in values]
        if any(not 0 <= value < 1 for value in momenta) or len(set(momenta)) != len(momenta):
            raise ValueError("momentum values must be unique and in [0, 1)")
        optim_base = {key: value for key, value in optimizer.items()
                      if key not in {"lr", "momentum", "nesterov"} and value is not None}
        optim_variants = []
        for momentum in momenta:
            variant = dict(optim_base)
            if momentum > 0:
                variant["momentum"] = momentum
                if nesterov is not None:
                    variant["nesterov"] = nesterov
            optim_variants.append(variant)
    else:
        optim_variants = [{key: value for key, value in optimizer.items()
                           if key != "lr" and value is not None}]
    resolved_fixed = dict(fixed)
    reg = resolved_fixed.get("reg")
    if isinstance(reg, dict) and "lam" in reg and reg["lam"] is None:
        resolved_fixed["reg"] = None
    regularizations = [resolved_fixed.get("reg")]
    if "reg" in search:
        reg_spec = search["reg"]
        if (not isinstance(reg_spec, dict) or not {"type", "lam"} <= set(reg_spec)
                or set(reg_spec) - {"type", "lam", "include_none"}
                or reg_spec["type"] not in ("l1", "l2")):
            raise ValueError("search.reg requires type (l1/l2), a lam range, "
                             "and optional include_none")
        include_none = reg_spec.get("include_none", False)
        if not isinstance(include_none, bool):
            raise ValueError("search.reg.include_none must be a boolean")
        regularizations = [{"type": reg_spec["type"], "lam": lam}
                           for lam in iter_values(reg_spec["lam"])]
        if any(reg["lam"] < 0 for reg in regularizations):
            raise ValueError("regularization strengths must be non-negative")
        if include_none:
            regularizations.insert(0, None)

    for depth in depths:
        for units in product(*widths[:depth]):
            for rate in rates:
                for optim in optim_variants:
                    for regularization in regularizations:
                        layers = [{"units": unit, "act": hidden_act} for unit in units]
                        layers.append({"units": output["units"], "act": output["activation"]})
                        yield {**resolved_fixed, "arch": layers, "reg": regularization,
                               "optim": {**optim, "lr": rate}}
