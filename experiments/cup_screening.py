import sys
import itertools
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.cup import load_cup_train, cup_design_test_split
from src.data.preprocessing import StandardScaler
from src.model_selection.build import build_model
from src.model_selection.grid_search import grid_search

N_INPUTS, N_OUTPUTS = 12, 4
INIT_FOR_ACT = {"relu": "he", "tanh": "glorot", "sigmoid": "glorot"}
EPOCHS_BUDGET = 2000

def make_arch(units, act, n_hidden=1):
    init = INIT_FOR_ACT[act]
    layers = [{"units": units, "act": act, "init": init} for _ in range(n_hidden)]
    layers.append({"units": N_OUTPUTS, "act": "identity", "init": "glorot"})
    return layers

def optimizer_blocks(coarse=True):
    blocks = []
    # --- SGD family ---
    scheds = [{"lr_schedule": "constant"}] if coarse else [
        {"lr_schedule": "constant"},
        {"lr_schedule": "linear", "lr_final_frac": 0.01},
    ]
    for lr in [0.05, 0.1]:
        for mom in [{"momentum": 0.0, "nesterov": False},
                    {"momentum": 0.9, "nesterov": False},
                    {"momentum": 0.9, "nesterov": True}]:
            for sc in scheds:
                if sc["lr_schedule"] == "linear":
                    actual_lr = {"type": "linear_decay", "eta_0": lr, "tau": EPOCHS_BUDGET, "eta_tau": lr * sc["lr_final_frac"]}
                else:
                    actual_lr = lr
                blocks.append({"type": "sgd", "lr": actual_lr, "momentum": mom["momentum"], "nesterov": mom["nesterov"]})
                
    # --- AdaGrad (adattivo: no momentum/schedule) ---
    for lr in [0.1, 0.5]:
        blocks.append({"type": "adagrad", "lr": lr})
        
    # --- QuickProp (full-batch, Newton-ish: no momentum/schedule) ---
    blocks.append({"type": "quickprop", "lr": 0.01, "mu": 1.75}) # Note: mu instead of mu_max per QuickProp signature
    
    return blocks

def build_coarse_grid():
    UNITS, ACTS, LAMBDAS = [20, 50], ["tanh", "relu"], [0.0, 1e-4, 1e-3]
    grid = {}
    
    configs = []
    for units, act, lam in itertools.product(UNITS, ACTS, LAMBDAS):
        arch = make_arch(units, act, n_hidden=1)
        for opt in optimizer_blocks(coarse=True):
            cfg = {
                "arch": arch, 
                "n_inputs": N_INPUTS, 
                "loss": "mse",
                "optim": opt,
                "epochs": EPOCHS_BUDGET,   # generoso: l'early stopping taglia
                "patience": 50,            # kfold.py expects patience at root level
                "min_delta": 0.0,
                "monitor": "val_mee",
                "scale": True
            }
            if lam > 0.0:
                cfg["reg"] = {"type": "l2", "lam": lam}
            
            # Create a string key to name the configs if we need to print them nicely, or just append
            configs.append(cfg)
            
    return configs

def main():
    print("="*50)
    print(" ML-CUP Coarse Grid Search (Phase 2)")
    print("="*50)
    
    _ids, X_full, Y_full = load_cup_train(str(ROOT / "data" / "cup" / "ML-CUP25-TR.csv"))
    X_tr, y_tr, _X_test, _y_test = cup_design_test_split(X_full, Y_full)
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # Manual scaling removed! We let kfold.py scale per-fold rigorously.
    
    param_grid = build_coarse_grid()
    print(f"Total configurations to test: {len(param_grid)}")
    
    # Since grid_search in our project expects a param_grid dict of lists (to do Cartesian product internally),
    # but build_coarse_grid ALREADY builds the flattened list of configs, we can't use iter_grid directly if we pass param_grid.
    # Wait, our grid_search function:
    # def grid_search(param_grid, build_model, X, Y, k=5, seed=None):
    # It does: configs = list(iter_grid(param_grid))
    # We should adapt grid_search to accept a pre-built list of configs, or map this to a param_grid.
    # Let's bypass iter_grid by just running the loop locally here since this script is bespoke for CUP.
    
    from src.model_selection.kfold import cross_validate
    
    results = []
    for i, cfg in enumerate(param_grid, 1):
        print(f"\n[{i}/{len(param_grid)}] Testing config:")
        print(f"  Arch: {cfg['arch'][0]['units']} {cfg['arch'][0]['act']}")
        print(f"  Optim: {cfg['optim']}")
        print(f"  Reg: {cfg.get('reg', 'None')}")
        
        metrics = cross_validate(build_model, cfg, X_tr, y_tr, k=3, seed=42, metric="mee") # k=3 for coarse screening speed
        
        result_entry = {
            'config': cfg,
            **metrics
        }
        results.append(result_entry)
        val_mee = metrics.get('val_mee_mean', 'N/A')
        print(f"  -> Result (mean val MEE): {val_mee}")

    print("\nGrid search done. Sorting results...")
    results.sort(key=lambda x: x.get('val_mee_mean', float('inf')))
    
    print("\n--- TOP 5 CONFIGURATIONS ---")
    for i in range(min(5, len(results))):
        c = results[i]['config']
        mean = results[i]['val_mee_mean']
        std = results[i]['val_mee_std']
        print(f"#{i+1}: MEE {mean:.4f} ± {std:.4f} | Arch: {c['arch'][0]['units']} {c['arch'][0]['act']} | Optim: {c['optim']} | Reg: {c.get('reg')}")
        
    out_dir = ROOT / "results" / "cup"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "coarse_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'coarse_grid_results.json'}")
if __name__ == "__main__":
    main()
