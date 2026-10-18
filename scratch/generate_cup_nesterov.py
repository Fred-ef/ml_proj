import json

archs = []

def make_layer(units, act):
    return {'units': units, 'act': act, 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}

def make_out():
    return {'units': 4, 'act': 'identity', 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}

# Riduciamo i nodi: 20, 40, 60, 80
for units in [20, 40, 60, 80]:
    # Fissiamo l'attivazione a relu (come nel modello 3)
    archs.append([make_layer(units, 'relu'), make_out()])

optim_grid = []
# Fissiamo il learning rate a 0.06 come nel modello 3
lr = 0.06
# Regoliamo il momentum normale e testiamo sia con Nesterov che senza Nesterov
for mom in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
    for nesterov in [True, False]:
        optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom, 'nesterov': nesterov})

# Fissiamo la regolarizzazione a 0.0009 come nel modello 3
reg_grid = [{'type': 'l2', 'lam': 0.0009}]

config = {
  'task': 'cup',
  'mode': 'select',
  'tag': 'grid-nesterov',
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

with open('configs/cup_grid_nesterov.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/cup_grid_nesterov.json with {len(archs) * len(optim_grid) * len(reg_grid)} configurations.")
