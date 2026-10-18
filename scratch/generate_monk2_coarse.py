import json

archs = []
activations = ['sigmoid', 'tanh']
init_scale = 0.3

def make_layer(units, act):
    return {'units': units, 'act': act, 'init': 'uniform', 'init_kwargs': {'scale': init_scale}}

def make_out():
    return {'units': 1, 'act': 'sigmoid', 'init': 'uniform', 'init_kwargs': {'scale': init_scale}}

# 1 hidden layer
for act in activations:
    for u1 in [4, 8, 16, 32, 64]:
        archs.append([make_layer(u1, act), make_out()])

# 2 hidden layers (decreasing units)
for act in activations:
    for u1 in [16, 32, 64]:
        for u2 in [8, 16, 32]:
            if u1 >= u2:
                archs.append([make_layer(u1, act), make_layer(u2, act), make_out()])

optim_grid = []
for lr in [0.1, 0.4, 0.7, 0.9]:
    for mom in [0.0, 0.5, 0.8, 0.9]:
        if mom == 0.0:
            optim_grid.append({'type': 'sgd', 'lr': lr})
        else:
            optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom})

config = {
  'task': 'monk2',
  'mode': 'select',
  'tag': 'grid-coarse-minibatch32',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 1000,
      'patience': 100,
      'min_delta': 1e-4,
      'batch_size': 32
    },
    'grid': {
      'arch': archs,
      'optim': optim_grid,
      'reg': [None]
    }
  }
}

with open('configs/monk2_grid_coarse_minibatch32.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk2_grid_coarse_minibatch32.json with {len(archs) * len(optim_grid)} configurations.")
