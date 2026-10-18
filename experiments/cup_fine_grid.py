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
from src.model_selection.kfold import cross_validate

N_INPUTS, N_OUTPUTS = 12, 4
INIT_FOR_ACT = {"relu": "he", "tanh": "glorot", "sigmoid": "glorot"}
EPOCHS_BUDGET = 2000

def make_arch(units, act, n_hidden=1):
    init = INIT_FOR_ACT[act]
    layers = [{"units": units, "act": act, "init": init} for _ in range(n_hidden)]
    layers.append({"units": N_OUTPUTS, "act": "identity", "init": "glorot"})
    return layers

def build_fine_grid():
    # Spingiamo i bordi: unità oltre 50, lambda oltre 1e-3
    UNITS = [50, 80, 120]
    ACTS = ["tanh"]
    LAMBDAS = [1e-3, 3e-3, 1e-2]
    
    configs = []
    
    # 1. Main SGD space
    for units, act, lam in itertools.product(UNITS, ACTS, LAMBDAS):
        arch = make_arch(units, act, n_hidden=1)
        
        scheds = [
            {"lr_schedule": "constant"},
            {"lr_schedule": "linear", "lr_final_frac": 0.01},
        ]
        
        for lr in [0.05, 0.1]:
            for mom in [{"momentum": 0.9, "nesterov": False},
                        {"momentum": 0.9, "nesterov": True}]:
                for sc in scheds:
                    if sc["lr_schedule"] == "linear":
                        actual_lr = {"type": "linear_decay", "eta_0": lr, "tau": EPOCHS_BUDGET, "eta_tau": lr * sc["lr_final_frac"]}
                    else:
                        actual_lr = lr
                    
                    cfg = {
                        "arch": arch, 
                        "n_inputs": N_INPUTS, 
                        "loss": "mse",
                        "optim": {"type": "sgd", "lr": actual_lr, "momentum": mom["momentum"], "nesterov": mom["nesterov"]},
                        "epochs": EPOCHS_BUDGET,
                        "patience": 50,
                        "min_delta": 0.0,
                        "monitor": "val_mee",
                        "scale": True,
                        "reg": {"type": "l2", "lam": lam}
                    }
                    configs.append(cfg)
                    
    # 2. Add Baseline configurations for report comparison (AdaGrad and QuickProp)
    # Using the safest "best" architecture and lambda from the coarse grid just to have a data point
    base_arch = make_arch(50, "tanh", n_hidden=1)
    base_lam = 1e-3
    
    adagrad_cfg = {
        "arch": base_arch, 
        "n_inputs": N_INPUTS, 
        "loss": "mse",
        "optim": {"type": "adagrad", "lr": 0.1},
        "epochs": EPOCHS_BUDGET,
        "patience": 50,
        "min_delta": 0.0,
        "monitor": "val_mee",
        "scale": True,
        "reg": {"type": "l2", "lam": base_lam}
    }
    configs.append(adagrad_cfg)
    
    quickprop_cfg = {
        "arch": base_arch, 
        "n_inputs": N_INPUTS, 
        "loss": "mse",
        "optim": {"type": "quickprop", "lr": 0.01, "mu": 1.75},
        "epochs": EPOCHS_BUDGET,
        "patience": 50,
        "min_delta": 0.0,
        "monitor": "val_mee",
        "scale": True,
        "reg": {"type": "l2", "lam": base_lam}
    }
    configs.append(quickprop_cfg)
            
    return configs

def main():
    print("="*50)
    print(" ML-CUP Fine Grid Search (Phase 3)")
    print("="*50)
    
    _ids, X_full, Y_full = load_cup_train(str(ROOT / "data" / "cup" / "ML-CUP25-TR.csv"))
    # Split out the internal test set, BUT DO NOT TOUCH IT AT THE END
    X_tr, y_tr, _X_test, _y_test = cup_design_test_split(X_full, Y_full)
    
    param_grid = build_fine_grid()
    print(f"Total configurations to test: {len(param_grid)}")
    
    results = []
    for i, cfg in enumerate(param_grid, 1):
        print(f"\n[{i}/{len(param_grid)}] Testing config:")
        print(f"  Arch: {cfg['arch'][0]['units']} {cfg['arch'][0]['act']}")
        print(f"  Optim: {cfg['optim']}")
        print(f"  Reg: {cfg.get('reg', 'None')}")
        
        # k=5 for statistical robustness in the fine grid
        metrics = cross_validate(build_model, cfg, X_tr, y_tr, k=5, seed=42, metric="mee") 
        
        result_entry = {
            'config': cfg,
            **metrics
        }
        results.append(result_entry)
        val_mee = metrics.get('val_mee_mean', 'N/A')
        val_mee_std = metrics.get('val_mee_std', 'N/A')
        print(f"  -> Result (mean val MEE): {val_mee} ± {val_mee_std}")

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
    with open(out_dir / "fine_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'fine_grid_results.json'}")

if __name__ == "__main__":
    main()
