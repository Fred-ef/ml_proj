import json

fixed_arch = [
  {
    "units": 4,
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

# Griglia molto fine attorno al punto vincente: lr=0.7, m=0.9
lrs = [0.6, 0.65, 0.7, 0.75, 0.8]
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

# Reintroduciamo una piccolissima regolarizzazione L2 per testare se abbatte ulteriormente eventuali microscopiche varianze
reg_grid = [
    None,
    {'type': 'l2', 'lam': 1e-5},
    {'type': 'l2', 'lam': 1e-4}
]

config = {
  'task': 'monk2',
  'mode': 'select',
  'tag': 'grid-fine-minibatch32',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 1000,
      'patience': 100,
      'min_delta': 1e-4,
      'batch_size': 32,
      'arch': fixed_arch
    },
    'grid': {
      'optim': optim_grid,
      'reg': reg_grid
    }
  }
}

with open('configs/monk2_grid_fine_minibatch32.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk2_grid_fine_minibatch32.json with {len(optim_grid) * len(reg_grid)} configurations.")
