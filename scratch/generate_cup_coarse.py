import json

archs = []
activations = ['relu', 'tanh']
init_scale = 0.3

def make_layer(units, act):
    return {'units': units, 'act': act, 'init': 'uniform', 'init_kwargs': {'scale': init_scale}}

def make_out():
    # Per CUP i target sono 4 valori continui -> output lineare (identity)
    return {'units': 4, 'act': 'identity', 'init': 'uniform', 'init_kwargs': {'scale': init_scale}}

# Generazione architetture (1 e 2 strati nascosti, come consigliato)
for act in activations:
    # 1 layer
    archs.append([make_layer(16, act), make_out()])
    archs.append([make_layer(32, act), make_out()])
    archs.append([make_layer(64, act), make_out()])
    archs.append([make_layer(128, act), make_out()])
    
    # 2 layers
    archs.append([make_layer(32, act), make_layer(16, act), make_out()])
    archs.append([make_layer(64, act), make_layer(32, act), make_out()])
    archs.append([make_layer(128, act), make_layer(64, act), make_out()])

optim_grid = []
for lr in [1e-4, 1e-3, 1e-2]:
    for mom in [0.6, 0.8, 0.9]:
        optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom})

# Esplorazione rigorosa della regolarizzazione su scala logaritmica (fondamentale per regressione)
reg_grid = [
    {'type': 'l2', 'lam': 1e-5},
    {'type': 'l2', 'lam': 1e-4},
    {'type': 'l2', 'lam': 1e-3},
    {'type': 'l2', 'lam': 1e-2}
]

config = {
  'task': 'cup',
  'mode': 'select',
  'tag': 'grid-coarse',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 1000,
      'patience': 50,  # Early stopping obbligatorio
      'min_delta': 0
    },
    'grid': {
      'arch': archs,
      'optim': optim_grid,
      'batch_size': [16, 32, 64, None],
      'reg': reg_grid
    }
  }
}

with open('configs/cup_grid_coarse.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/cup_grid_coarse.json with {len(archs) * len(optim_grid) * 4 * len(reg_grid)} configurations.")
