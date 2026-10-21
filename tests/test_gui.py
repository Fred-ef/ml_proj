"""Smoke tests for service/gui.py: the NiceGUI console mounted on the same
FastAPI app as service/app.py.

Not a UI test (no browser) — just confirms the mount doesn't break anything:
the console route serves HTML, and the original API routes (already covered
in depth by test_service.py) are still reachable through the SAME app object.

Run from the project root:  pytest tests/test_gui.py -v
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from nicegui import ui

from service.gui import SelectForm, app

client = TestClient(app)


def test_console_page_serves_html():
    r = client.get("/gui/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_console_mount_path_without_trailing_slash_also_resolves():
    assert client.get("/gui").status_code == 200


def test_original_api_routes_still_reachable_through_the_mounted_app():
    assert client.get("/docs").status_code == 200
    assert client.get("/jobs").status_code == 200
    assert client.get("/runs/monk1/does-not-exist").status_code == 404


def test_select_form_carries_n_trials_through_build_and_load():
    seen = {}

    @ui.page("/_select_form_probe")
    def probe():
        # NiceGUI elements exist only inside a page, so the form is exercised from one.
        form = SelectForm()
        seen["default"] = form.build()["n_trials"]
        form.load({"n_trials": 5, "fixed": {}, "grid": {}})
        seen["loaded"] = form.build()["n_trials"]
        form.load({"fixed": {}, "grid": {}})     # a template saved before n_trials existed
        seen["template"] = form.build()["n_trials"]

    assert client.get("/gui/_select_form_probe").status_code == 200
    assert seen == {"default": 3, "loaded": 5, "template": 1}
