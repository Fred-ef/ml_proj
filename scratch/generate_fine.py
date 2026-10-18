import json

archs = []
activations = ['tanh']
init_scales = [0.2, 0.3, 0.4]

def make_layer(units, act, scale):
    return {'units': units, 'act': act, 'init': 'uniform', 'init_kwargs': {'scale': scale}}

def make_out(scale):
    return {'units': 1, 'act': 'sigmoid', 'init': 'uniform', 'init_kwargs': {'scale': scale}}

# Fissiamo l'architettura vincente (32->16->8->4) e la seconda classificata (32->8->8)
# variando solo lo scale di inizializzazione
for scale in init_scales:
    # Top 1: 32-16-8-4
    archs.append([
        make_layer(32, 'tanh', scale),
        make_layer(16, 'tanh', scale),
        make_layer(8, 'tanh', scale),
        make_layer(4, 'tanh', scale),
        make_out(scale)
    ])
    # Top 2: 32-8-8
    archs.append([
        make_layer(32, 'tanh', scale),
        make_layer(8, 'tanh', scale),
        make_layer(8, 'tanh', scale),
        make_out(scale)
    ])

optim_grid = []
# Visto che il best era lr=0.7 e momentum=0.8, esploriamo l'intorno fine
for lr in [0.6, 0.7, 0.8]:
    for mom in [0.7, 0.8, 0.9]:
        optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom})

config = {
  'task': 'monk1',
  'mode': 'select',
  'tag': 'grid-fine-v2',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 1000,
      'patience': 50,
      'batch_size': None
    },
    'grid': {
      'arch': archs,
      'optim': optim_grid,
      'reg': [None]
    }
  }
}

with open('configs/monk1_grid_fine.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk1_grid_fine.json with {len(archs) * len(optim_grid)} configurations.")
