import json

archs = []

# Inizializzazione standard e sicura
def make_layer(units, act):
    return {'units': units, 'act': act, 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}

# Output rigorosamente lineare (Identity) per la regressione
def make_out():
    return {'units': 4, 'act': 'identity', 'init': 'uniform', 'init_kwargs': {'scale': 0.3}}

# Vincolo: max 2 layer, nodi tra 10 e 50
topologies = [
    # 1 layer nascosto (tra 10 e 50)
    [10], [20], [30], [40], [50],
    
    # 2 layer nascosti (strutture a imbuto entro i limiti)
    [20, 10], 
    [30, 10], [30, 20], 
    [40, 20], [40, 30],
    [50, 20], [50, 30]
]

# Funzioni adatte alla regressione (nei layer nascosti):
# ReLU per evitare saturazione, Tanh per gestire bene distribuzioni continue simmetriche
activations = ['relu', 'tanh']

for topo in topologies:
    for act in activations:
        layers = []
        for units in topo:
            layers.append(make_layer(units, act))
        layers.append(make_out())
        archs.append(layers)

config = {
  'task': 'cup',
  'mode': 'select',
  'tag': 'grid-topo-small',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 2000,
      'patience': 50,       # Early stopping libero (min_delta: 0)
      'min_delta': 0,
      'batch_size': 32,     # Fissato al valore dell'assessment precedente
      
      # "Il resto disattivato" -> Niente accelerazione né regolarizzazione
      'optim': {
        'type': 'sgd',
        'lr': 0.0001,       # Learning rate "standard" sicuro che abbiamo usato
        'momentum': 0.0,    # Disattivato
        'nesterov': False   # Disattivato
      },
      'reg': None           # Disattivato
    },
    'grid': {
      'arch': archs
    }
  }
}

with open('configs/cup_grid_topo_small.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/cup_grid_topo_small.json with {len(archs)} configurations.")
