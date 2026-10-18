import json

archs = []

def make_layer(units, act):
    return {'units': units, 'act': act, 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}

def make_out():
    return {'units': 4, 'act': 'identity', 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}

# Assestiamoci intorno a 50 unità
for units in [40, 50, 60]:
    archs.append([make_layer(units, 'relu'), make_out()])

optim_grid = []
# Learning rate fisso al migliore trovato prima (0.06)
lr = 0.06
# Modifichiamo il momentum stringendo intorno al top 0.95
for mom in [0.85, 0.9, 0.95, 0.98]:
    for nesterov in [True, False]:
        optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom, 'nesterov': nesterov})

reg_grid = []
# Modifichiamo la regolarizzazione stringendo intorno al top 0.0009
for lam in [0.0003, 0.0006, 0.0009, 0.0012, 0.0015]:
    reg_grid.append({'type': 'l2', 'lam': lam})

config = {
  'task': 'cup',
  'mode': 'select',
  'tag': 'grid-fine-50units',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 30000,
      'patience': 200,      
      'min_delta': 0.009,   
      'batch_size': None,   # Full-Batch
      'scale': True         # Z-score scaler
    },
    'grid': {
      'arch': archs,
      'optim': optim_grid,
      'reg': reg_grid
    }
  }
}

with open('configs/cup_grid_fine_50units.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/cup_grid_fine_50units.json with {len(archs) * len(optim_grid) * len(reg_grid)} configurations.")
