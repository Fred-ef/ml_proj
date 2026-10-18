import json

# Fissiamo l'ottimizzatore a un buon default trovato in precedenza
optim_fixed = {'type': 'sgd', 'lr': 0.5, 'momentum': 0.8}

# Helper functions
def make_layer(units, act, init_strategy):
    # init_strategy is a tuple: (init_name, kwargs)
    layer = {'units': units, 'act': act, 'init': init_strategy[0]}
    if init_strategy[1]:
        layer['init_kwargs'] = init_strategy[1]
    return layer

def make_out(init_strategy):
    layer = {'units': 1, 'act': 'sigmoid', 'init': init_strategy[0]}
    if init_strategy[1]:
        layer['init_kwargs'] = init_strategy[1]
    return layer

# Definiamo le architetture chiave da esplorare (forma a imbuto o flat)
arch_shapes = [
    # 1 Hidden
    [4], [8], [16], [32],
    # 2 Hidden
    [16, 8], [32, 16], [32, 8], [16, 16],
    # 3 Hidden
    [32, 16, 8], [16, 16, 8], [32, 8, 4],
    # 4 Hidden
    [32, 16, 8, 4]
]

# Definiamo le combinazioni logiche di Attivazione + Inizializzazione
# Non ha senso testare he con sigmoid, o glorot con relu. Testiamo le accoppiate giuste.
act_init_pairs = [
    # Tanh e Sigmoid vanno a braccetto con Glorot o Uniforme (scale piccole)
    ('tanh', ('glorot', {})),
    ('tanh', ('uniform', {'scale': 0.1})),
    ('tanh', ('uniform', {'scale': 0.3})),
    
    ('sigmoid', ('glorot', {})),
    ('sigmoid', ('uniform', {'scale': 0.1})),
    ('sigmoid', ('uniform', {'scale': 0.3})),
    
    # ReLU va a braccetto con He o Uniforme
    ('relu', ('he', {})),
    ('relu', ('uniform', {'scale': 0.1})),
    ('relu', ('uniform', {'scale': 0.3}))
]

archs_grid = []

# Prodotto cartesiano Architetture x (Attivazione+Init)
for shape in arch_shapes:
    for act, init_strat in act_init_pairs:
        current_arch = []
        for nodes in shape:
            current_arch.append(make_layer(nodes, act, init_strat))
        
        # Aggiungiamo l'output layer (sempre sigmoid per la classificazione!)
        # Usiamo per coerenza la stessa inizializzazione dei layer nascosti
        current_arch.append(make_out(init_strat))
        archs_grid.append(current_arch)

config = {
  'task': 'monk1',
  'mode': 'select',
  'tag': 'stage1-topology',
  'config': {
    'k': 3,
    'seed': 42,
    'fixed': {
      'loss': 'mse',
      'epochs': 500,  # Fissiamo a 500 senza early stopping per dare a tutte tempo di convergere
      'batch_size': None
    },
    'grid': {
      'arch': archs_grid,
      'optim': [optim_fixed],
      'reg': [None]
    }
  }
}

with open('configs/monk1_stage1.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Generated configs/monk1_stage1.json with {len(archs_grid)} topological configurations.")
