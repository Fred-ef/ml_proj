import json

# Architettura vincente della Fase 1
best_arch = [
    {'units': 32, 'act': 'relu', 'init': 'uniform', 'init_kwargs': {'scale': 0.3}},
    {'units': 8,  'act': 'relu', 'init': 'uniform', 'init_kwargs': {'scale': 0.3}},
    {'units': 1,  'act': 'sigmoid', 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}
]

optim_grid = []

# 1. SGD (Costante vs LinearDecay, Momentum, Nesterov)
lrs = [0.1, 0.3, 0.5, 0.7, 0.9]
momentums = [0.0, 0.5, 0.8, 0.9]
nesterovs = [False, True]

for lr in lrs:
    for mom in momentums:
        for nest in nesterovs:
            # Constant LR
            optim_grid.append({
                'type': 'sgd', 'lr': lr, 'momentum': mom, 'nesterov': nest
            })
            # Linear Decay LR (decays to 0.01 * lr over 300 steps)
            optim_grid.append({
                'type': 'sgd',
                'lr': {'type': 'linear_decay', 'eta_0': lr, 'tau': 300},
                'momentum': mom,
                'nesterov': nest
            })

# 2. AdaGrad
for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:
    optim_grid.append({'type': 'adagrad', 'lr': lr})

# 3. QuickProp (only works with full batch, which is set to None below)
for lr in [0.01, 0.1, 0.5]:
    for mu in [1.5, 1.75, 2.0]:
        optim_grid.append({'type': 'quickprop', 'lr': lr, 'mu': mu})

config = {
  'task': 'monk1',
  'mode': 'select',
  'tag': 'stage2-optim',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,
      'batch_size': None, # Full batch is required for QuickProp
      'arch': best_arch
    },
    'grid': {
      'optim': optim_grid,
      'reg': [None]
    }
  }
}

with open('configs/monk1_stage2.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk1_stage2.json with {len(optim_grid)} optimizer configurations.")
