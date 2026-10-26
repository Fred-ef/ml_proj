"""Build a Network from a config dict

Every call creates new weights and a new optimizer, so each fold or trial
starts from a clean state
"""

from ..nn.layer import Dense
from ..nn.network import Network
from ..nn.activations import Identity, Sigmoid, Tanh, ReLU
from ..nn.initializers import Uniform, Glorot, He
from ..nn.losses import MSE
from ..nn.optimizers import SGD
from ..nn.rates import LinearDecay
from ..nn.regularizers import L2, L1

# config strings -> classes
_ACT  = {"identity": Identity, "sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU}
_INIT = {"uniform": Uniform, "glorot": Glorot, "he": He}
_LOSS = {"mse": MSE}
_OPT  = {"sgd": SGD}
_REG  = {"l2": L2, "l1": L1}
_LR_SCHEDULES = {"linear_decay": LinearDecay}


def build_model(config: dict) -> Network:
    """Costruisce una Network FRESCA (pesi + optimizer nuovi) da una config"""
    n_in = config["n_inputs"]
    layers = []
    for spec in config["arch"]:
        act = _ACT[spec["act"]]()                                   
        init_name   = spec.get("init", config.get("init", "uniform"))
        init_kwargs = spec.get("init_kwargs")
        if init_kwargs is None:
            init_kwargs = config.get("init_kwargs") or {}
        init = _INIT[init_name](**init_kwargs)                     
        layers.append(Dense(n_in, spec["units"], act, init))
        n_in = spec["units"]                                        

    loss = _LOSS[config.get("loss", "mse")]()

    opt_cfg  = dict(config["optim"]); opt_type = opt_cfg.pop("type")
    
    if "lr" in opt_cfg and isinstance(opt_cfg["lr"], dict):
        lr_cfg = dict(opt_cfg["lr"])
        lr_type = lr_cfg.pop("type")
        opt_cfg["lr"] = _LR_SCHEDULES[lr_type](**lr_cfg)
        
    optimizer = _OPT[opt_type](**opt_cfg)

    reg = None
    if config.get("reg"):
        reg_cfg = dict(config["reg"]); reg_type = reg_cfg.pop("type")
        reg = _REG[reg_type](**reg_cfg)

    return Network(layers, loss=loss, optimizer=optimizer, regularizer=reg,
                   seed=config.get("seed"), shuffle_seed=config.get("shuffle_seed"))
