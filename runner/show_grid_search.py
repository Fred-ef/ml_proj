"""Read a saved grid search summary without running the search again.

Usage:
    python -m runner.show_grid_search results/monk1/RUN_ID
    python -m runner.show_grid_search results/monk1/RUN_ID/summary.json
    python -m runner.show_grid_search results/monk1/RUN_ID --top 20
    python -m runner.show_grid_search results/monk1/RUN_ID --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple


class _ConfigField(NamedTuple):
    key: str
    label: str
    display: str
    raw: object


def _number(value: float) -> str:
    return f"{value:.6g}"


def _value(value) -> str:
    """Render nested configuration values without JSON punctuation."""
    if value is None:
        return "nessuno"
    if isinstance(value, bool):
        return "sì" if value else "no"
    if isinstance(value, float):
        return str(value)
    if isinstance(value, dict):
        kind = str(value.get("type", ""))
        params = ", ".join(f"{key}={_value(item)}" for key, item in value.items()
                           if key != "type")
        return f"{kind} ({params})" if kind and params else kind or params or "nessuno"
    if isinstance(value, list):
        return "; ".join(_value(item) for item in value)
    return str(value)


def _architecture(config: dict) -> str:
    layers = config.get("arch", [])
    parts = [f"{config.get('n_inputs', '?')} input"]
    parts.extend(f"{layer.get('units', '?')} ({layer.get('act', '?')})"
                 for layer in layers)
    return " -> ".join(parts)


def _init_kwargs(config: dict, layer: dict) -> dict:
    kwargs = layer.get("init_kwargs")
    return (config.get("init_kwargs") or {}) if kwargs is None else kwargs


def _initialization(config: dict) -> str:
    specs = []
    for layer in config.get("arch", []):
        kind = layer.get("init", config.get("init", "uniform"))
        kwargs = _init_kwargs(config, layer)
        details = ", ".join(f"{key}={_value(value)}" for key, value in kwargs.items())
        specs.append(f"{kind} ({details})" if details else kind)
    if not specs:
        return "non specificata"
    if all(spec == specs[0] for spec in specs):
        return f"{specs[0]} in tutti i layer"
    return "; ".join(f"layer {index}: {spec}" for index, spec in enumerate(specs, 1))


def _batch_mode(batch_size) -> str:
    if batch_size is None:
        return "full batch (1 aggiornamento per epoca)"
    if batch_size == 1:
        return "per campione (batch size 1)"
    return f"batch di {batch_size} campioni per aggiornamento"


def _configuration_fields(config: dict) -> list[_ConfigField]:
    """Keep a stable, human-readable order for each configuration's parameters."""
    layers = config.get("arch", [])
    architecture = (config.get("n_inputs"),
                    tuple((layer.get("units"), layer.get("act")) for layer in layers))
    initialization = tuple((layer.get("init", config.get("init", "uniform")),
                            _init_kwargs(config, layer))
                           for layer in layers)
    patience = config.get("patience")
    fields = [
        _ConfigField("arch", "Architettura", _architecture(config), architecture),
        _ConfigField("init", "Inizializzazione", _initialization(config), initialization),
        _ConfigField("loss", "Loss di addestramento",
                     str(config.get("loss", "mse")).upper(), config.get("loss", "mse")),
        _ConfigField("epochs", "Epoche massime", _value(config.get("epochs", 100)),
                     config.get("epochs", 100)),
        _ConfigField("batch_size", "Aggiornamento pesi",
                     _batch_mode(config.get("batch_size")), config.get("batch_size")),
    ]
    optim = config.get("optim", {})
    if isinstance(optim, dict):
        fields.append(_ConfigField("optim.type", "Ottimizzatore",
                                   str(optim.get("type", "?")).upper(), optim.get("type")))
        opt_labels = {"lr": "Learning rate", "momentum": "Momentum",
                      "nesterov": "Nesterov", "epsilon": "Epsilon", "mu": "Mu"}
        fields.extend(_ConfigField(f"optim.{key}",
                                   opt_labels.get(key, key.replace("_", " ").capitalize()),
                                   _value(value), value)
                      for key, value in optim.items() if key != "type")
    else:
        fields.append(_ConfigField("optim", "Ottimizzatore", _value(optim), optim))

    reg = config.get("reg")
    if isinstance(reg, dict):
        fields.append(_ConfigField("reg.type", "Regolarizzazione",
                                   str(reg.get("type", "?")).upper(), reg.get("type")))
        fields.extend(_ConfigField(f"reg.{key}",
                                   "Lambda" if key == "lam" else key.replace("_", " ").capitalize(),
                                   _value(value), value)
                      for key, value in reg.items() if key != "type")
    else:
        fields.append(_ConfigField("reg", "Regolarizzazione", "nessuna", reg))
    fields.append(_ConfigField("scale", "Standardizzazione",
                               "input e target" if config.get("scale", False)
                               else "disattivata", config.get("scale", False)))
    min_delta = config.get("min_delta") or 0.0
    stopping = ("disattivato" if patience is None else
                f"patience={patience}, min_delta={_value(min_delta)}")
    fields.append(_ConfigField("early_stopping", "Early stopping", stopping,
                               (patience, min_delta if patience is not None else None)))

    known = {"n_inputs", "arch", "init", "init_kwargs", "loss", "epochs",
             "batch_size", "optim",
             "reg", "scale", "patience", "min_delta"}
    fields.extend(_ConfigField(key, key.replace("_", " ").capitalize(),
                               _value(config[key]), config[key])
                  for key in sorted(config.keys() - known))
    return fields


def _append_fields(lines: list[str], fields: list[tuple[str, str]], indent: str) -> None:
    if not fields:
        lines.append(f"{indent}nessuna")
        return
    width = max(len(label) for label, _ in fields)
    lines.extend(f"{indent}{label:<{width}} : {value}" for label, value in fields)


def _same_raw(left, right) -> bool:
    """Compare JSON configuration values without rounding or key-order effects."""
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _runs_label(row: dict, validation_mode: str) -> str:
    """Name the runs behind the epoch lists ('' = a single holdout run)."""
    trials = row.get("n_trials", 1) > 1
    if validation_mode == "kfold":
        return "nei run (trial × fold)" if trials else "nei fold"
    return "nei trial" if trials else ""


def _ranking_table(ranking: list[dict], limit: int | None) -> list[str]:
    """One line per configuration: mean error +- std across initializations."""
    if not ranking or "n_trials" not in ranking[0]:
        return []  # summaries saved before multi-initialization support
    rows = ranking if limit is None else ranking[:limit]

    def cell(mean, std) -> str:
        return _number(mean) if std is None else f"{_number(mean)} ± {_number(std)}"

    n_trials = ranking[0]["n_trials"]
    spread = (f"errore medio ± dev. std tra {n_trials} inizializzazioni dei pesi"
              if n_trials > 1 else "una sola inizializzazione: nessuna dev. std")
    lines = ["", f"CLASSIFICA SINTETICA ({spread})",
             f"  {'#':>3}  {'Validation':<24} {'Training':<24} {'Epoca':>6}"]
    for rank, row in enumerate(rows, start=1):
        lines.append(f"  {rank:>3}  {cell(row['val_error'], row['val_init_std']):<24} "
                     f"{cell(row['train_error'], row['train_init_std']):<24} "
                     f"{row['best_epoch_median']:>6}")
    return lines


def _init_fields(row: dict, metric_label: str) -> list[tuple[str, str]]:
    """Spread caused by the initial weights (absent from older summaries)."""
    n_trials = row.get("n_trials", 1)
    if n_trials < 2:
        return []
    fields = [("Inizializzazioni dei pesi (trial)", str(n_trials))]
    for name, label in (("val", "validation"), ("train", "training")):
        fields.append((f"Varianza errore {label} (tra inizializzazioni)",
                       _number(row[f"{name}_init_var"])))
        fields.append((f"Dev. std errore {label} (tra inizializzazioni)",
                       _number(row[f"{name}_init_std"])))
    fields.append((f"Validation per trial ({metric_label})",
                   ", ".join(_number(value) for value in row["per_trial_val"])))
    return fields


def _epoch_fields(row: dict, validation_mode: str) -> list[tuple[str, str]]:
    """Distinguish training duration from the best validation epoch.

    Older summaries recorded only the validation-optimal epoch, not training
    duration. Their labels must not imply that scores came from the final epoch.
    """
    if "epochs_run" not in row:
        label = ("Mediana epoche di miglior validation (criterio precedente)"
                 if validation_mode == "kfold"
                 else "Epoca di miglior validation (criterio precedente)")
        fields = [(label, str(row["best_epoch_median"]))]
        if validation_mode == "kfold" and "best_epochs" in row:
            epochs = ", ".join(str(epoch) for epoch in row["best_epochs"])
            fields.append(("Epoche di miglior validation nei fold", epochs))
        return fields

    epochs_run = row["epochs_run"]
    where = _runs_label(row, validation_mode)
    if where:
        completed = ", ".join(str(epoch) for epoch in epochs_run)
        fields = [(f"Epoche eseguite {where} (fine training)", completed)]
    else:
        fields = [("Epoche eseguite (fine training)", str(epochs_run[0]))]

    if row["config"].get("patience") is not None:
        if where:
            selected = ", ".join(str(epoch) for epoch in row["best_epochs"])
            fields.append((f"Epoche di migliore validation {where}", selected))
            fields.append(("Mediana epoche di migliore validation",
                           str(row["best_epoch_median"])))
        else:
            fields.append(("Epoca di migliore validation",
                           str(row["best_epochs"][0])))
    return fields


def format_select_summary(summary: dict, run_config: dict | None = None,
                          limit: int | None = 20) -> str:
    """Show shared settings and the first ranked configurations."""
    metric = summary["metric"]
    metric_label = {"acc": "accuratezza", "mee": "MEE"}.get(metric, metric)
    order = {"acc": "accuratezza di validation decrescente",
             "mee": "MEE di validation crescente"}.get(
                 metric, "errore di validation crescente")
    task = f"{run_config['task'].upper()}  |  " if run_config and "task" in run_config else ""
    lines = [
        "RISULTATI GRID SEARCH",
        f"  {task}Validazione: {summary['validation_mode']}  |  "
        f"Configurazioni: {summary['n_configs']}  |  Metrica: {metric_label}",
        "",
    ]

    ranking = summary["ranking"]
    lines.extend(_ranking_table(ranking, limit))
    if any("epochs_run" not in row for row in ranking):
        lines.extend((
            "",
            "NOTA SUI RISULTATI PRECEDENTI",
            "  Per le configurazioni senza durata registrata, le metriche sono",
            "  quelle dell'epoca di miglior validation (per ogni fold nella k-fold),",
            "  anche senza early stopping. La fine del training non è registrata.",
        ))
    shown = ranking if limit is None else ranking[:limit]
    described = [_configuration_fields(row["config"]) for row in ranking]
    other_maps = [{field.key: field.raw for field in fields} for fields in described[1:]]
    common = [field for field in described[0]
              if all(field.key in other and _same_raw(other[field.key], field.raw)
                     for other in other_maps)] if described else []
    common_keys = {field.key for field in common}
    run_fields = []
    if run_config:
        if summary["validation_mode"] == "holdout":
            fraction = run_config.get("val_fraction")
            fraction = 0.2 if fraction is None else fraction
            run_fields.append(("Quota validation", f"{_number(100 * fraction)}%"))
        elif run_config.get("k") is not None:
            run_fields.append(("Numero di fold", str(run_config["k"])))
        if "seed" in run_config:
            run_fields.append(("Seed", _value(run_config["seed"])))
    lines.extend(("", "IMPOSTAZIONI COMUNI"))
    _append_fields(lines, run_fields + [(field.label, field.display) for field in common],
                   "  ")

    heading = f"CONFIGURAZIONI ORDINATE ({order})"
    if len(shown) < len(ranking):
        heading += f" - prime {len(shown)} di {len(ranking)}"
    lines.extend(("", heading))
    for rank, row in enumerate(shown, start=1):
        lines.extend(("", "-" * 72, f"CONFIGURAZIONE {rank}/{summary['n_configs']}",
                      "  Risultati:"))
        error_suffix = " (MEE media)" if metric == "mee" else ""
        results = [
            (f"Errore training{error_suffix}", _number(row["train_error"])),
            (f"Errore validation{error_suffix}", _number(row["val_error"])),
        ]
        if metric == "acc":
            results.extend((("Accuratezza training", _number(row["train_mean"])),
                            ("Accuratezza validation", _number(row["val_mean"]))))
        elif metric not in ("mee", "loss"):
            results.extend(((f"{metric_label} training", _number(row["train_mean"])),
                            (f"{metric_label} validation", _number(row["val_mean"]))))
        for name, label in (("train_std", "training"), ("val_std", "validation")):
            std = row[name]
            if std is not None:
                results.append((f"Deviazione standard {metric_label} {label} (tra fold)",
                                _number(std)))
        results.extend(_init_fields(row, metric_label))
        results.extend(_epoch_fields(row, summary["validation_mode"]))
        _append_fields(lines, results, "    ")
        lines.append("  Iperparametri specifici:")
        specific = [(field.label, field.display) for field in described[rank - 1]
                    if field.key not in common_keys]
        if specific:
            _append_fields(lines, specific, "    ")
        else:
            lines.append("    nessun parametro variabile (vedi impostazioni comuni)")
    if len(shown) < len(ranking):
        remaining = len(ranking) - len(shown)
        noun = "configurazione" if remaining == 1 else "configurazioni"
        lines.extend(("", f"Risultati restanti: {remaining} {noun} in "
                      "summary.json; usa --all per mostrarli."))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Display a saved grid search result.")
    parser.add_argument("result", type=Path, help="run directory or summary.json path")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--top", type=int, metavar="N",
                           help="show the first N configurations (default: 20)")
    selection.add_argument("--all", action="store_true",
                           help="show every configuration")
    args = parser.parse_args(argv)
    if args.top is not None and args.top < 1:
        parser.error("--top must be a positive integer")
    path = args.result / "summary.json" if args.result.is_dir() else args.result
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    required = {"metric", "validation_mode", "n_configs", "ranking"}
    if not isinstance(summary, dict) or not required <= summary.keys():
        parser.error(f"{path} is not a grid search summary")
    run_config = None
    config_path = path.parent / "config.json"
    if config_path.is_file():
        try:
            run_config = json.loads(config_path.read_text(encoding="utf-8")).get("config")
        except (OSError, json.JSONDecodeError):
            pass
    print(format_select_summary(summary, run_config=run_config,
                                limit=None if args.all else (args.top or 20)))
    curves_dir = path.parent / "curves"
    if curves_dir.is_dir():
        print(f"\nCurve medie (loss a sinistra, metrica a destra; ± dev. std tra inizializzazioni): "
              f"{curves_dir}/rankNN.png"
              f"\nConfronto delle validation migliori: {curves_dir / 'top_val.png'}")


if __name__ == "__main__":
    main()
