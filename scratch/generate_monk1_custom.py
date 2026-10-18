import json

# Unità nascoste per reti shallow (1 solo hidden layer)
hidden_units = [2, 3, 4, 5, 10]

# Tipi di attivazione e inizializzazione che sappiamo funzionare bene
act_inits = [
    ('relu', 'uniform', {'scale': 0.3}),
    ('tanh', 'glorot', {})
]

archs_grid = []
for units in hidden_units:
    for act, init, init_kwargs in act_inits:
        arch = [
            {'units': units, 'act': act, 'init': init},
            {'units': 1, 'act': 'sigmoid', 'init': init}
        ]
        if init_kwargs:
            arch[0]['init_kwargs'] = init_kwargs
            arch[1]['init_kwargs'] = init_kwargs
        archs_grid.append(arch)

# Learning Rate e Momentum specificati
lrs = [0.1, 0.3, 0.5, 0.7]
momentums = [0.5, 0.7, 0.9]

optim_grid = []
for lr in lrs:
    for mom in momentums:
        optim_grid.append({
            'type': 'sgd',
            'lr': lr,
            'momentum': mom
        })

config = {
  'task': 'monk1',
  'mode': 'select',
  'tag': 'custom-compact',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,
      'batch_size': None # Batch gradient descent
    },
    'grid': {
      'arch': archs_grid,
      'optim': optim_grid,
      'reg': [None] # Nessuna regolarizzazione (lambda=0)
    }
  }
}

with open('configs/monk1_custom.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk1_custom.json with {len(archs_grid) * len(optim_grid)} configurations.")
