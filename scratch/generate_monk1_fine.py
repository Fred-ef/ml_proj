import json

# Unità nascoste esplorate finemente attorno al 10
hidden_units = [6, 8, 10, 12]

archs_grid = []
for units in hidden_units:
    arch = [
        {"units": units, "act": "tanh", "init": "glorot"},
        {"units": 1, "act": "sigmoid", "init": "glorot"}
    ]
    archs_grid.append(arch)

# Griglia 2D accoppiata per LR e Momentum
lrs = [0.4, 0.5, 0.6, 0.7]
momentums = [0.5, 0.6, 0.7, 0.8]

optim_grid = []
for lr in lrs:
    for mom in momentums:
        optim_grid.append({
            'type': 'sgd',
            'lr': lr,
            'momentum': mom
        })

# Regolarizzazione L2 micro-dosata per ridurre la varianza
reg_grid = [
    None,
    {'type': 'l2', 'lam': 1e-5},
    {'type': 'l2', 'lam': 1e-4}
]

config = {
  'task': 'monk1',
  'mode': 'select',
  'tag': 'stage3-fine-guidelines',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,
      'patience': 50,      # Attiviamo l'early stopping su validazione!
      'batch_size': None
    },
    'grid': {
      'arch': archs_grid,
      'optim': optim_grid,
      'reg': reg_grid
    }
  }
}

with open('configs/monk1_stage3_fine.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk1_stage3_fine.json with {len(archs_grid) * len(optim_grid) * len(reg_grid)} configurations.")
