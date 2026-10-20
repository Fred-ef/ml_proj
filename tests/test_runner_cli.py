"""Tests for the command-line runner's result display."""

import sys
from unittest.mock import Mock

import pytest

from runner import __main__ as cli


@pytest.mark.parametrize("mode", ["select", "train"])
def test_cli_displays_saved_grid_search_only_for_select(monkeypatch, tmp_path, capsys, mode):
    run_dir = tmp_path / "run-1"
    payload = {"seed": 7}
    load_config = Mock(return_value={
        "task": "monk1", "mode": mode, "tag": "smoke", "config": payload,
    })
    run_experiment = Mock(return_value=run_dir)
    show_summary = Mock()
    monkeypatch.setattr(cli, "load_config", load_config)
    monkeypatch.setattr(cli, "run_experiment", run_experiment)
    monkeypatch.setattr(cli.show_grid_search, "main", show_summary)
    monkeypatch.setattr(sys, "argv", ["runner", "--config", "smoke.json"])
    monkeypatch.chdir(tmp_path)

    cli.main()

    load_config.assert_called_once_with("smoke.json")
    run_experiment.assert_called_once_with("monk1", mode, payload, tag="smoke")
    output = capsys.readouterr().out
    assert output.startswith(f"Risultati salvati: {run_dir.name}\n")
    assert output.count(run_dir.name) == 1
    assert '"run_id"' not in output
    if mode == "select":
        show_summary.assert_called_once_with([str(run_dir)])
    else:
        show_summary.assert_not_called()
