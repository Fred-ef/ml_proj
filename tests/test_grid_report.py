"""The saved grid result should be readable without inspecting JSON syntax."""

import json

from runner.show_grid_search import format_select_summary, main


def _reported_value(report: str, label: str) -> str:
    line = next(line for line in report.splitlines() if line.strip().startswith(label))
    return line.rsplit(":", 1)[1].strip()


def test_saved_report_shows_common_settings_and_complete_config_blocks(tmp_path, capsys):
    base = {
        "n_inputs": 17,
        "arch": [{"units": 2, "act": "tanh", "init": "uniform",
                  "init_kwargs": {"scale": 0.3}},
                 {"units": 1, "act": "sigmoid", "init": "uniform",
                  "init_kwargs": {"scale": 0.3}}],
        "loss": "mse", "epochs": 12, "batch_size": None,
        "reg": None,
    }
    summary = {
        "metric": "acc", "validation_mode": "holdout", "n_configs": 2,
        "ranking": [
            {"config": {**base, "optim": {"type": "sgd", "lr": 0.1, "momentum": 0.9}},
             "train_error": 0.1, "val_error": 0.2, "train_mean": 0.9,
             "val_mean": 0.8, "train_std": None, "val_std": None,
             "best_epoch_median": 5},
            {"config": {**base, "optim": {"type": "sgd", "lr": 0.01, "momentum": 0.9}},
             "train_error": 0.2, "val_error": 0.3, "train_mean": 0.8,
             "val_mean": 0.7, "train_std": None, "val_std": None,
             "best_epoch_median": 4},
        ],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"config": {
        "task": "monk1", "seed": 0, "val_fraction": 0.25,
    }}), encoding="utf-8")

    main([str(tmp_path)])
    report = capsys.readouterr().out

    assert "MEDIA E VARIANZA DEGLI ERRORI TRA CONFIGURAZIONI" not in report
    assert report.index("IMPOSTAZIONI COMUNI") < report.index("CONFIGURAZIONE 1/2")
    assert "Quota validation" in report and "25%" in report
    assert "17 input -> 2 (tanh) -> 1 (sigmoid)" in report
    assert "full batch (1 aggiornamento per epoca)" in report
    assert report.count("CONFIGURAZIONE ") == 2
    assert report.count("Learning rate") == 2  # changes between configurations
    assert report.count("Momentum") == 1       # shown only among common settings
    assert report.count("Architettura") == 1
    assert report.count("Epoche massime") == 1
    assert report.count("Aggiornamento pesi") == 1
    assert report.count("Iperparametri specifici:") == 2
    assert "Deviazione standard" not in report
    assert report.count("Epoca di miglior validation (criterio precedente)") == 2
    assert "NOTA SUI RISULTATI PRECEDENTI" in report
    assert "anche senza early stopping" in report
    assert "Epoche eseguite" not in report
    assert "0.1" in report and "0.01" in report
    assert "{" not in report and '"arch"' not in report

    summary["ranking"][0]["config"]["optim"]["lr"] = 0.12345671
    summary["ranking"][1]["config"]["optim"]["lr"] = 0.12345679
    precise_report = format_select_summary(summary)
    assert "0.12345671" in precise_report
    assert "0.12345679" in precise_report
    assert precise_report.count("Learning rate") == 2

    single = {**summary, "ranking": summary["ranking"][:1], "n_configs": 1}
    single_report = format_select_summary(single)
    assert "nessun parametro variabile (vedi impostazioni comuni)" in single_report

    kfold = {**single, "validation_mode": "kfold",
             "ranking": [{**single["ranking"][0], "train_std": 0.02,
                          "val_std": 0.03}]}
    kfold_report = format_select_summary(kfold)
    assert "Deviazione standard accuratezza training" in kfold_report
    assert "Deviazione standard accuratezza validation" in kfold_report
    assert "Mediana epoche di miglior validation (criterio precedente)" in kfold_report


def test_report_ignores_legacy_aggregate_and_points_to_curves(tmp_path, capsys):
    row = {"config": {"n_inputs": 1, "arch": [{"units": 1, "act": "sigmoid"}],
                      "optim": {"type": "sgd", "lr": 0.1}, "epochs": 3},
           "train_error": 0.1, "val_error": 0.2, "train_mean": 0.9, "val_mean": 0.8,
           "train_std": None, "val_std": None,
           "best_epoch_median": 3, "best_epochs": [3], "epochs_run": [3]}
    summary = {"metric": "acc", "validation_mode": "holdout", "n_configs": 1,
               "aggregate": {name: {"mean": 0.1, "variance": 0.0}
                             for name in ("train_error", "val_error")},
               "ranking": [row]}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    main([str(tmp_path)])
    report = capsys.readouterr().out
    assert "MEDIA E VARIANZA DEGLI ERRORI TRA CONFIGURAZIONI" not in report
    assert "CONFIGURAZIONE 1/1" in report
    assert "Curve medie" not in report                  # no curves/ folder yet

    (tmp_path / "curves").mkdir()
    main([str(tmp_path)])
    report = capsys.readouterr().out
    assert "Curve medie" in report and "top_val.png" in report


def test_report_distinguishes_completed_epochs_from_best_validation_epochs():
    config = {
        "n_inputs": 1,
        "arch": [{"units": 1, "act": "sigmoid"}],
        "optim": {"type": "sgd", "lr": 0.1},
        "epochs": 12,
    }
    row = {
        "config": config,
        "train_error": 0.1, "val_error": 0.2,
        "train_mean": 0.9, "val_mean": 0.8,
        "train_std": None, "val_std": None,
        "epochs_run": [12],
        "best_epoch_median": 4,
        "best_epochs": [4],
    }
    summary = {
        "metric": "acc", "validation_mode": "holdout", "n_configs": 1,
        "ranking": [row],
    }

    holdout = format_select_summary(summary)
    assert _reported_value(holdout, "Epoche eseguite (fine training)") == "12"
    assert "criterio precedente" not in holdout
    assert "NOTA SUI RISULTATI PRECEDENTI" not in holdout
    assert "Epoca di migliore validation" not in holdout

    row["config"] = {**config, "patience": 3}
    row["epochs_run"] = [9]
    stopped_holdout = format_select_summary(summary)
    assert _reported_value(stopped_holdout, "Epoche eseguite (fine training)") == "9"
    assert _reported_value(stopped_holdout, "Epoca di migliore validation") == "4"

    summary["validation_mode"] = "kfold"
    row["epochs_run"] = [9, 10, 11]
    row["best_epochs"] = [4, 6, 7]
    row["best_epoch_median"] = 6
    stopped_kfold = format_select_summary(summary)
    assert _reported_value(stopped_kfold,
                           "Epoche eseguite nei fold (fine training)") == "9, 10, 11"
    assert _reported_value(stopped_kfold,
                           "Epoche di migliore validation nei fold") == "4, 6, 7"
    assert _reported_value(stopped_kfold,
                           "Mediana epoche di migliore validation") == "6"

    row["config"] = config
    row["epochs_run"] = [12, 12, 12]
    plain_kfold = format_select_summary(summary)
    assert _reported_value(plain_kfold,
                           "Epoche eseguite nei fold (fine training)") == "12, 12, 12"
    assert "Epoche di migliore validation nei fold" not in plain_kfold
    assert "criterio precedente" not in plain_kfold


def test_report_shows_first_twenty_and_all_option_restores_full_list(tmp_path, capsys):
    ranking = []
    for rank in range(1, 22):
        ranking.append({
            "config": {"n_inputs": 1, "arch": [{"units": 1, "act": "sigmoid"}],
                       "optim": {"type": "sgd", "lr": 0.1 if rank <= 20 else 0.01}},
            "train_error": rank / 100,
            "val_error": rank / 100,
            "train_mean": 1 - rank / 100,
            "val_mean": 1 - rank / 100,
            "train_std": None,
            "val_std": None,
            "best_epoch_median": rank,
        })
    summary = {
        "metric": "acc", "validation_mode": "holdout", "n_configs": 21,
        "ranking": ranking,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    main([str(tmp_path)])
    report = capsys.readouterr().out
    assert report.count("CONFIGURAZIONE ") == 20
    assert "CONFIGURAZIONE 20/21" in report
    assert "CONFIGURAZIONE 21/21" not in report
    assert "prime 20 di 21" in report
    assert "Risultati restanti: 1 configurazione in summary.json" in report
    assert report.count("Learning rate") == 20  # last row still affects common settings

    main([str(tmp_path), "--top", "5"])
    top_five = capsys.readouterr().out
    assert top_five.count("CONFIGURAZIONE ") == 5
    assert "CONFIGURAZIONE 5/21" in top_five

    main([str(tmp_path), "--all"])
    full_report = capsys.readouterr().out
    assert full_report.count("CONFIGURAZIONE ") == 21
    assert "CONFIGURAZIONE 21/21" in full_report
