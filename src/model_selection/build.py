"""Model factory: build a Network from a plain config dict.

Maps the string keys in a config (e.g. "tanh", "sgd", "l2") to the
corresponding classes and assembles a new Network — new weights and a new
optimizer on every call — each k-fold / trial starts from a clean state
"""

from ..nn.layer import Dense
from ..nn.network import Network
from ..nn.activations import Identity, Sigmoid, Tanh, ReLU
from ..nn.initializers import Uniform, Glorot, He
from ..nn.losses import MSE, CrossEntropy
from ..nn.optimizers import SGD, QuickProp
from ..nn.regularizers import L2, L1
from ..nn.rates import AdaGrad, LinearDecay

# Dispatch: config strings -> classes
_ACT  = {"identity": Identity, "sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU}
_INIT = {"uniform": Uniform, "glorot": Glorot, "he": He}
_LOSS = {"mse": MSE, "cross_entropy": CrossEntropy}
_OPT  = {"sgd": SGD, "quickprop": QuickProp, "adagrad": AdaGrad}
_REG  = {"l2": L2, "l1": L1}
_LR_SCHEDULES = {"linear_decay": LinearDecay}


def build_model(config: dict) -> Network:
    """Costruisce una Network FRESCA (pesi + optimizer nuovi) da una config."""
    n_in = config["n_inputs"]
    if not isinstance(n_in, int) or n_in <= 0:
        raise ValueError("n_inputs must be a positive integer")
    if not config.get("arch"):
        raise ValueError("arch must contain at least one layer")
    layers = []
    for spec in config["arch"]:
        if not isinstance(spec.get("units"), int) or spec["units"] <= 0:
            raise ValueError("every layer must have a positive integer number of units")
        act_name = spec.get("act")
        if act_name not in _ACT:
            raise ValueError(f"unknown activation {act_name!r}; expected one of {sorted(_ACT)}")
        act = _ACT[act_name]()
        init_name   = spec.get("init", config.get("init", "uniform"))
        init_kwargs = spec.get("init_kwargs", {})
        if init_name not in _INIT:
            raise ValueError(f"unknown initializer {init_name!r}; expected one of {sorted(_INIT)}")
        try:
            init = _INIT[init_name](**init_kwargs)
        except TypeError as exc:
            raise ValueError(f"invalid arguments for initializer {init_name!r}: {exc}") from exc
        layers.append(Dense(n_in, spec["units"], act, init))
        n_in = spec["units"]                                        

    loss_name = config.get("loss", "mse")
    if loss_name not in _LOSS:
        raise ValueError(f"unknown loss {loss_name!r}; expected one of {sorted(_LOSS)}")
    loss = _LOSS[loss_name]()

    if "optim" not in config or not isinstance(config["optim"], dict):
        raise ValueError("optim must be a configuration object")
    opt_cfg = dict(config["optim"])
    opt_type = opt_cfg.pop("type", None)
    if opt_type not in _OPT:
        raise ValueError(f"unknown optimizer {opt_type!r}; expected one of {sorted(_OPT)}")
    
    if "lr" in opt_cfg and isinstance(opt_cfg["lr"], dict):
        lr_cfg = dict(opt_cfg["lr"])
        lr_type = lr_cfg.pop("type", None)
        if lr_type not in _LR_SCHEDULES:
            raise ValueError(
                f"unknown learning-rate schedule {lr_type!r}; "
                f"expected one of {sorted(_LR_SCHEDULES)}"
            )
        opt_cfg["lr"] = _LR_SCHEDULES[lr_type](**lr_cfg)
        
    optimizer = _OPT[opt_type](**opt_cfg)

    reg = None
    if config.get("reg"):
        reg_cfg = dict(config["reg"]); reg_type = reg_cfg.pop("type", None)
        if reg_type not in _REG:
            raise ValueError(f"unknown regularizer {reg_type!r}; expected one of {sorted(_REG)}")
        reg = _REG[reg_type](**reg_cfg)

    return Network(layers, loss=loss, optimizer=optimizer, regularizer=reg, seed=config.get("seed"))
