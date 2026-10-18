import json

fixed_arch = [
  {
    "units": 2,
    "act": "sigmoid",
    "init": "uniform",
    "init_kwargs": {
      "scale": 0.3
    }
  },
  {
    "units": 1,
    "act": "sigmoid",
    "init": "uniform",
    "init_kwargs": {
      "scale": 0.3
    }
  }
]

# Griglia molto fine attorno al Modello 3 scelto (lr=0.6, m=0.9)
lrs = [0.5, 0.6, 0.7]
momentums = [0.8, 0.85, 0.9, 0.95]
nesterovs = [False, True]

optim_grid = []
for lr in lrs:
    for mom in momentums:
        for nest in nesterovs:
            optim_grid.append({
                'type': 'sgd',
                'lr': lr,
                'momentum': mom,
                'nesterov': nest
            })

# Il Modello 4 usava Reg:- (Nessuna regolarizzazione). Esploriamo da None fino a 1e-3.
reg_grid = [
    None,
    {'type': 'l2', 'lam': 1e-5},
    {'type': 'l2', 'lam': 1e-4},
    {'type': 'l2', 'lam': 5e-4},
    {'type': 'l2', 'lam': 1e-3}
]

config = {
  'task': 'monk3',
  'mode': 'select',
  'tag': 'grid-fine-fullbatch',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,
      'patience': 30,
      'min_delta': 1e-4,
      'batch_size': None,
      'arch': fixed_arch
    },
    'grid': {
      'optim': optim_grid,
      'reg': reg_grid
    }
  }
}

with open('configs/monk3_grid_fine_fullbatch.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk3_grid_fine_fullbatch.json with {len(optim_grid) * len(reg_grid)} configurations.")
