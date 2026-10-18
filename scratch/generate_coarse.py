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
    for u1 in [2, 4, 8, 16, 32]:
        archs.append([make_layer(u1, act), make_out()])

# 2 hidden layers (decreasing units)
for act in activations:
    for u1 in [8, 16, 32]:
        for u2 in [4, 8, 16]:
            if u1 >= u2:
                archs.append([make_layer(u1, act), make_layer(u2, act), make_out()])

# 3 hidden layers
for act in activations:
    for u1 in [16, 32]:
        for u2 in [8, 16]:
            for u3 in [4, 8]:
                if u1 >= u2 >= u3:
                    archs.append([make_layer(u1, act), make_layer(u2, act), make_layer(u3, act), make_out()])

# 4 hidden layers (total 5 layers counting output)
for act in activations:
    for u1 in [32]:
        for u2 in [16]:
            for u3 in [8]:
                for u4 in [4]:
                    if u1 >= u2 >= u3 >= u4:
                        archs.append([make_layer(u1, act), make_layer(u2, act), make_layer(u3, act), make_layer(u4, act), make_out()])

optim_grid = []
for lr in [0.1, 0.4, 0.7, 0.9]:
    for mom in [0.0, 0.5, 0.8, 0.9]:
        if mom == 0.0:
            optim_grid.append({'type': 'sgd', 'lr': lr})
        else:
            optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom})

config = {
  'task': 'monk1',
  'mode': 'select',
  'tag': 'grid-coarse',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,
      'batch_size': None
    },
    'grid': {
      'arch': archs,
      'optim': optim_grid,
      'reg': [None]
    }
  }
}

with open('configs/monk1_grid_coarse.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk1_grid_coarse.json with {len(archs)} architectures.")
