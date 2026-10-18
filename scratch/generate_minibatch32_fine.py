import json

archs = [
    # Modello 1 (Rete Profonda)
    [
        {"units": 32, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
        {"units": 16, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
        {"units": 8, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
        {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}
    ],
    # Modello 3 (Rete Piccola)
    [
        {"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
        {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}
    ]
]

# Uniamo i due intorni per il learning rate: attorno a 0.4 e attorno a 0.9
lrs = [0.3, 0.4, 0.5, 0.8, 0.9, 0.95]
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

reg_grid = [
    None,
    {'type': 'l2', 'lam': 1e-5},
    {'type': 'l2', 'lam': 1e-4}
]

config = {
  'task': 'monk1',
  'mode': 'select',
  'tag': 'stage3-fine-minibatch32',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,
      'patience': 50,
      'min_delta': 1e-4,
      'batch_size': 32
    },
    'grid': {
      'arch': archs,
      'optim': optim_grid,
      'reg': reg_grid
    }
  }
}

with open('configs/monk1_grid_fine_minibatch32.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk1_grid_fine_minibatch32.json with {len(archs) * len(optim_grid) * len(reg_grid)} configurations.")
