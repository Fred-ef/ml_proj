import numpy as np
from src.utils.metrics import accuracy, mse
from src.model_selection.build import build_model          # la factory di §6.1

def run_trials(config: dict, X_tr, y_tr, X_te, y_te, metrics: dict) -> tuple[dict, list[dict]]:
    """Esegue config su n_trials seed diversi; ritorna (summary, histories)."""
    n_trials  = config.get("n_trials", 5)
    base_seed = config.get("seed", 0)

    test_acc, train_acc, histories = [], [], []
    for i in range(n_trials):
        cfg_i = {**config, "seed": base_seed + i}
        model = build_model(cfg_i)

        history = model.fit(X_tr, y_tr, epochs=cfg_i["epochs"],
                            batch_size=cfg_i.get("batch_size"),
                            validation_data=(X_te, y_te), metrics=metrics)
                            
        pred_tr, pred_te = model.predict(X_tr), model.predict(X_te)
        acc_tr, acc_te = accuracy(pred_tr, y_tr), accuracy(pred_te, y_te)
        train_acc.append(acc_tr); test_acc.append(acc_te); histories.append(history)

    test_acc, train_acc = np.array(test_acc), np.array(train_acc)
    rep = int(np.argsort(test_acc)[len(test_acc) // 2])

    summary = {
        "test_acc_mean":  float(np.mean(test_acc)),
        "test_acc_std":   float(np.std(test_acc)),
        "train_acc_mean": float(np.mean(train_acc)),
        "train_acc_std":  float(np.std(train_acc)),
        "per_trial_test_acc": [float(a) for a in test_acc],   # tracciabilità completa
        "n_trials": n_trials,
        "representative_trial": rep,
    }
    return summary, histories  # plottqi histories[rep]