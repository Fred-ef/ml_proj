"""Model factory: build a Network from a plain config dict.

Maps the string keys in a config (e.g. "tanh", "sgd", "l2") to the
corresponding classes and assembles a new Network — new weights and a new
optimizer on every call — each k-fold / trial starts from a clean state
"""

from ..nn.layer import Dense
from ..nn.network import Network
from ..nn.activations import Identity, Sigmoid, Tanh, ReLU
from ..nn.initializers import Uniform, Glorot, He
from ..nn.losses import MSE
from ..nn.optimizers import SGD, QuickProp
from ..nn.rates import AdaGrad, LinearDecay
from ..nn.regularizers import L2, L1
from ..nn.rates import AdaGrad, LinearDecay

# Dispatch: config strings -> classes
_ACT  = {"identity": Identity, "sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU}
_INIT = {"uniform": Uniform, "glorot": Glorot, "he": He}
_LOSS = {"mse": MSE}
_OPT  = {"sgd": SGD, "quickprop": QuickProp, "adagrad": AdaGrad}
_REG  = {"l2": L2, "l1": L1}
_LR_SCHEDULES = {"linear_decay": LinearDecay}


def _resolve_lr(lr):
    if isinstance(lr, dict):
        sched_cfg = dict(lr); sched_type = sched_cfg.pop("type")
        return _LR_SCHEDULES[sched_type](**sched_cfg)
    return lr

def build_model(config: dict) -> Network:
    """Costruisce una Network FRESCA (pesi + optimizer nuovi) da una config."""
    n_in = config["n_inputs"]
    layers = []
    for spec in config["arch"]:
        act = _ACT[spec["act"]]()                                   
        init_name   = spec.get("init", config.get("init", "uniform"))
        init_kwargs = spec.get("init_kwargs", {})
        init = _INIT[init_name](**init_kwargs)                     
        layers.append(Dense(n_in, spec["units"], act, init))
        n_in = spec["units"]                                        

    loss = _LOSS[config.get("loss", "mse")]()

    opt_cfg  = dict(config["optim"]); opt_type = opt_cfg.pop("type")
    
    if "lr" in opt_cfg and isinstance(opt_cfg["lr"], dict):
        lr_cfg = dict(opt_cfg["lr"])
        lr_type = lr_cfg.pop("type")
        opt_cfg["lr"] = _LR_SCHEDULERS[lr_type](**lr_cfg)
        
    optimizer = _OPT[opt_type](**opt_cfg)

    reg = None
    if config.get("reg"):
        reg_cfg = dict(config["reg"]); reg_type = reg_cfg.pop("type")
        reg = _REG[reg_type](**reg_cfg)

    return Network(layers, loss=loss, optimizer=optimizer, regularizer=reg, seed=config.get("seed"))