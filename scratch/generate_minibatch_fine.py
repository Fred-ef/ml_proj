import json

fixed_arch = [
  {
    "units": 4,
    "act": "tanh",
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

# Proviamo ad azzerare ancora di più la varianza
reg_grid = [
    None,
    {'type': 'l2', 'lam': 1e-5},
    {'type': 'l2', 'lam': 1e-4}
]

config = {
  'task': 'monk1',
  'mode': 'select',
  'tag': 'stage3-fine-minibatch',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,
      'patience': 50,
      'min_delta': 1e-4,
      'batch_size': 16,
      'arch': fixed_arch
    },
    'grid': {
      'optim': optim_grid,
      'reg': reg_grid
    }
  }
}

with open('configs/monk1_stage3_fine_minibatch.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk1_stage3_fine_minibatch.json with {len(optim_grid) * len(reg_grid)} configurations.")
