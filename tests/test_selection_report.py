"""Presentation of selection rankings, including single-split results."""

from runner.selection_report import render_selection_report


def test_accuracy_ranking_is_descending_with_clear_fields_and_no_fake_std():
    def candidate(score, units, lr):
        return {
            "val_mean": score, "val_std": None, "best_epoch_median": 12,
            "config": {
                "loss": "mse", "epochs": 50, "batch_size": None,
                "arch": [
                    {"units": units, "act": "tanh", "init": "glorot"},
                    {"units": 1, "act": "sigmoid", "init": "glorot"},
                ],
                "optim": {"type": "sgd", "lr": lr}, "reg": None,
            },
        }

    summary = {
        "metric": "acc", "selection_method": "holdout", "validation_fraction": 0.2,
        "ranking": [candidate(0.6, 2, 0.1), candidate(0.8, 4, 0.00001)],
    }
    report = render_selection_report(summary, n_inputs=17)
    first = report.index("1. mean val acc: 0.8000 (80.0%)")
    second = report.index("2. mean val acc: 0.6000 (60.0%)")
    assert first < second
    assert "17 input -> hidden 1: 4 (tanh, glorot)" in report
    assert "Optimizer:    sgd; lr=1e-05" in report
    assert "batch_size=full" in report
    assert "Validation std: not available from one split" in report
    assert "std: 0.0000" not in report


def test_lower_is_better_metric_is_sorted_ascending():
    summary = {
        "metric": "mee", "selection_method": "kfold",
        "ranking": [
            {"val_mean": 2.0, "best_epoch_median": 3, "config": {}},
            {"val_mean": 1.0, "best_epoch_median": 2, "config": {}},
        ],
    }
    report = render_selection_report(summary)
    assert report.index("1. mean val mee: 1.0000") < report.index("2. mean val mee: 2.0000")
