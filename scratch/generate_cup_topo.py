import json

archs = []

def make_layer(units, act, init, init_kwargs):
    return {'units': units, 'act': act, 'init': init, 'init_kwargs': init_kwargs}

def make_out(init, init_kwargs):
    # Output lineare fisso per regressione CUP (4 targets)
    return {'units': 4, 'act': 'identity', 'init': init, 'init_kwargs': init_kwargs}

# Forme topologiche: Espansione esplorativa verso reti molto grandi e profonde
topologies = [
    # Baseline (i vecchi vincitori da battere)
    [64, 32],
    [64, 32, 16],
    
    # Topologie grandi a 2 layer
    [64, 64],
    [128, 32],
    [128, 64],
    
    # Topologie enormi a 3 layer
    [64, 32, 32],
    [64, 64, 32],
    [128, 64, 32],
    
    # Topologie ultra-profonde a 4 layer
    [64, 32, 16, 8],
    [128, 64, 32, 16]
]

# Funzioni di attivazione per i layer nascosti
activations = ['relu', 'tanh']

# Strategie di inizializzazione pesi
initializers = [
    ('uniform', {'scale': 0.3}),
    ('uniform', {'scale': 0.1}),
    ('glorot', {}),   # Ideale teoricamente per Tanh
    ('he', {})        # Ideale teoricamente per ReLU
]

for topo in topologies:
    for act in activations:
        for init, kwargs in initializers:
            layers = []
            for units in topo:
                layers.append(make_layer(units, act, init, kwargs))
            layers.append(make_out(init, kwargs))
            archs.append(layers)

# Fissiamo gli iperparametri numerici sulla "safe zone" trovata finora
config = {
  'task': 'cup',
  'mode': 'select',
  'tag': 'grid-topo',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 2000,
      'patience': 50,
      'min_delta': 0,
      'batch_size': 16,
      'optim': {
        'type': 'sgd',
        'lr': 0.0001,
        'momentum': 0.0,
        'nesterov': False
      },
      'reg': None
    },
    'grid': {
      'arch': archs
    }
  }
}

with open('configs/cup_grid_topo.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/cup_grid_topo.json with {len(archs)} topological configurations.")
