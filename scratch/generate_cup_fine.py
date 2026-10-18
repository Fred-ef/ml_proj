import json

init_scale = 0.3

def make_layer(units):
    return {'units': units, 'act': 'relu', 'init': 'uniform', 'init_kwargs': {'scale': init_scale}}

def make_out():
    return {'units': 4, 'act': 'identity', 'init': 'uniform', 'init_kwargs': {'scale': init_scale}}

# Esplorazione chirurgica della capacità (16 o 24 neuroni)
archs = [
    [make_layer(16), make_out()],
    [make_layer(24), make_out()]
]

optim_grid = []
# Learning rate attorno a 1e-4, con opzione prudente per contenere oscillazioni
for lr in [5e-5, 1e-4, 5e-4]:
    for mom in [0.8, 0.9]:
        optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom})

# L2 torna a discriminare pulito grazie all'Early Stopping
reg_grid = [
    {'type': 'l2', 'lam': 1e-4},
    {'type': 'l2', 'lam': 1e-3},
    {'type': 'l2', 'lam': 1e-2}
]

config = {
  'task': 'cup',
  'mode': 'select',
  'tag': 'grid-fine',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 2000,       # Lasciamo massima libertà temporale
      'patience': 30,       # Tagliola stretta per fermarsi ai minimi
      'min_delta': 0,       # Zero tolleranza fittizia per non uccidere la curva
      'batch_size': 16      # Batch size vincente fisso
    },
    'grid': {
      'arch': archs,
      'optim': optim_grid,
      'reg': reg_grid
    }
  }
}

with open('configs/cup_grid_fine.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/cup_grid_fine.json with {len(archs) * len(optim_grid) * len(reg_grid)} configurations.")
