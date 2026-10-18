import json

archs = []

def make_layer(units, act):
    # Uso uniform(0.3) come default consolidato, a meno che non fosse specificato altro
    return {'units': units, 'act': act, 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}

def make_out():
    return {'units': 4, 'act': 'identity', 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}

# "# units" -> Assumo 1 singolo layer nascosto con queste capacità
for units in [20, 40, 80, 100, 150]:
    for act in ['relu', 'tanh']:  # Testiamo entrambe le attivazioni standard
        archs.append([make_layer(units, act), make_out()])

optim_grid = []
# Learning rate (eta) e momentum (alpha)
for lr in [0.03, 0.06, 0.08, 0.098]:
    for mom in [0.2, 0.4, 0.6, 0.8, 0.9]:
        optim_grid.append({'type': 'sgd', 'lr': lr, 'momentum': mom, 'nesterov': False})

reg_grid = []
# Regolarizzazione L2 (lambda)
for lam in [0.0001, 0.0005, 0.0009, 0.0013]:
    reg_grid.append({'type': 'l2', 'lam': lam})

config = {
  'task': 'cup',
  'mode': 'select',
  'tag': 'grid-coarse-table',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 30000,
      'patience': 200,      # s_UP = 200
      'min_delta': 0.009,   # epsilon_GL = 0.009
      'batch_size': 32,   # Full-Batch per reggere gli ETA altissimi
      'scale': True         # Attiviamo il vostro StandardScaler!
    },
    'grid': {
      'arch': archs,
      'optim': optim_grid,
      'reg': reg_grid
    }
  }
}

with open('configs/cup_grid_coarse_table.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/cup_grid_coarse_table.json with {len(archs) * len(optim_grid) * len(reg_grid)} configurations.")
