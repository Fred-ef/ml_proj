import json

archs = []
activations = ['sigmoid', 'tanh']
init_scale = 0.3

def make_layer(units, act):
    return {'units': units, 'act': act, 'init': 'uniform', 'init_kwargs': {'scale': init_scale}}

def make_out():
    return {'units': 1, 'act': 'sigmoid', 'init': 'uniform', 'init_kwargs': {'scale': init_scale}}

# 1 hidden layer (piccole e medie)
for act in activations:
    for u1 in [2, 4, 8, 16, 32]:
        archs.append([make_layer(u1, act), make_out()])

optim_grid = []
for lr in [0.3, 0.6, 0.9]:
    for mom in [0.6, 0.8, 0.9]:
        optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom})

reg_grid = [
    None,
    {'type': 'l2', 'lam': 1e-4},
    {'type': 'l2', 'lam': 1e-3}
]

config = {
  'task': 'monk3',
  'mode': 'select',
  'tag': 'grid-coarse-minibatch32',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,
      'patience': 30,
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

with open('configs/monk3_grid_coarse_minibatch32.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk3_grid_coarse_minibatch32.json with {len(archs) * len(optim_grid) * len(reg_grid)} configurations.")
