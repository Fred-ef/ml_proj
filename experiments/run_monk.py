import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.monk import load_monk
from src.utils.metrics import accuracy
from src.utils.plotting import plot_learning_curve
from src.utils.experiment import new_run_dir, save_run, append_index, index_to_csv, flatten
from _common import run_trials

CONFIG = {
    "task": "monk1", "n_inputs": 17,
    "arch": [{"units": 4, "act": "tanh",    "init": "uniform", "init_kwargs": {"scale": 0.3}},
             {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}],
    "loss": "mse", "optim": {"type": "sgd", "lr": 0.1, "momentum": 0.9}, "reg": None,
    "epochs": 300, "batch_size": None, "seed": 0, "n_trials": 5,
    "tag": "baseline",
}

def main() -> None:
    X_tr, y_tr = load_monk(str(ROOT / "data" / "monk" / "monks-1.train"))
    X_te, y_te = load_monk(str(ROOT / "data" / "monk" / "monks-1.test"))

    summary, histories = run_trials(CONFIG, X_tr, y_tr, X_te, y_te, metrics={"acc": accuracy})

    run_dir = new_run_dir(ROOT / "results", CONFIG["task"], CONFIG["tag"], CONFIG["seed"])
    rep = summary["representative_trial"]
    save_run(run_dir, CONFIG, summary, history=histories[rep])
    plot_learning_curve(histories[rep], metrics=("loss", "acc"),
                        save_path=str(run_dir / "learning_curve.png"),
                        title=f"{CONFIG['task']} — {CONFIG['tag']}")

    index = ROOT / "results" / CONFIG["task"] / "index.jsonl"
    row = {"run_id": run_dir.name,
           "arch": "-".join([str(CONFIG["n_inputs"])] + [str(l["units"]) for l in CONFIG["arch"]]),
           **flatten({k: CONFIG[k] for k in ("optim", "epochs")}),
           **summary}
    append_index(index, row); index_to_csv(index)

    print(f"Test acc: {summary['test_acc_mean']*100:.2f}% ± {summary['test_acc_std']*100:.2f}%")
    print(f"Saved to {run_dir}")

if __name__ == "__main__":
    main()