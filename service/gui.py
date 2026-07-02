"""NiceGUI console (mounted under the FastAPI app)

Run from the project root:
    uvicorn service.gui:app     # HTTP API + console at /gui
    uvicorn service.app:app     # HTTP API only
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional, get_args

from nicegui import ui

from runner.registry import TASKS, get_task
from service import gui_client as api
from service.app import app
from service.schemas import Act, Init, RegT

ROOT = Path(__file__).resolve().parents[1]
_CONFIGS_DIR = ROOT / "configs"


# --------------------------------------------------------------------- helpers
def _fmt_value(v: Any) -> Any:
    return f"{v:.4g}" if isinstance(v, float) else v


def _clock(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


def _load_templates() -> list[dict]:
    """configs/*.json examples, tagged with their source path so the Submit
    tab's dropdown can offer them and reload them verbatim (see README/configs)."""
    templates = []
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        data["_path"] = str(path)
        templates.append(data)
    return templates


_TEMPLATES = _load_templates()


# ---------------------------------------------------------------- form pieces
class LayerRow:
    """One row of `arch`. `scale` is visible only for init="uniform" — every
    other initializer takes zero kwargs (src/nn/initializers.py), so hiding it
    is what makes sending a stray kwarg structurally impossible from this UI."""

    def __init__(self, container: ui.column, on_remove, *, units: int = 4, act: str = "tanh",
                 init: str = "uniform", scale: float = 0.1) -> None:
        with container, ui.row().classes("items-center gap-2") as self.row:
            self.units = ui.number("units", value=units, min=1, precision=0).classes("w-20")
            self.act = ui.select(list(get_args(Act)), label="act", value=act).classes("w-28")
            self.init = ui.select(list(get_args(Init)), label="init", value=init).classes("w-24")
            self.scale = ui.number("scale", value=scale, min=0.0, step=0.01).classes("w-20")
            self.scale.bind_visibility_from(self.init, "value", backward=lambda v: v == "uniform")
            ui.button(icon="delete", on_click=lambda: on_remove(self)).props("flat dense round")

    def to_dict(self) -> dict:
        d = {"units": int(self.units.value or 1), "act": self.act.value, "init": self.init.value}
        d["init_kwargs"] = {"scale": float(self.scale.value or 0.1)} if self.init.value == "uniform" else {}
        return d

    def delete(self) -> None:
        self.row.delete()


class ArchSection:
    def __init__(self) -> None:
        self.rows: list[LayerRow] = []
        self.container = ui.column().classes("gap-1 w-full")
        ui.button("+ layer", icon="add", on_click=lambda: self.add_row()).props("outline dense")
        self.add_row()

    def add_row(self, **kw: Any) -> None:
        self.rows.append(LayerRow(self.container, self._remove, **kw))

    def _remove(self, row: LayerRow) -> None:
        if len(self.rows) <= 1:
            ui.notify("Serve almeno un layer", type="warning")
            return
        row.delete()
        self.rows.remove(row)

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.rows]

    def load(self, arch: list[dict]) -> None:
        for r in list(self.rows):
            r.delete()
        self.rows.clear()
        for layer in arch:
            kwargs = layer.get("init_kwargs") or {}
            self.add_row(units=layer["units"], act=layer["act"],
                        init=layer.get("init", "uniform"), scale=kwargs.get("scale", 0.1))


class OptimSection:
    """sgd and quickprop take disjoint kwargs (service/schemas.py's discriminated
    union) — the two field groups are mutually exclusive here for the same reason."""

    def __init__(self) -> None:
        with ui.row().classes("items-end gap-2"):
            self.type = ui.select(["sgd", "quickprop"], label="optimizer", value="sgd").classes("w-32")
            self.lr = ui.number("lr", value=0.1, min=0.0, step=0.01).classes("w-24")
            self.momentum = ui.number("momentum", value=0.9, min=0.0, max=1.0, step=0.05).classes("w-28")
            self.nesterov = ui.checkbox("nesterov")
            self.mu = ui.number("mu", value=1.75, min=0.0, step=0.05).classes("w-24")
        self.momentum.bind_visibility_from(self.type, "value", backward=lambda v: v == "sgd")
        self.nesterov.bind_visibility_from(self.type, "value", backward=lambda v: v == "sgd")
        self.mu.bind_visibility_from(self.type, "value", backward=lambda v: v == "quickprop")

    def to_dict(self) -> dict:
        if self.type.value == "quickprop":
            return {"type": "quickprop", "lr": float(self.lr.value or 0.1), "mu": float(self.mu.value or 1.75)}
        return {"type": "sgd", "lr": float(self.lr.value or 0.1),
                "momentum": float(self.momentum.value or 0.0), "nesterov": bool(self.nesterov.value)}

    def load(self, optim: dict) -> None:
        t = optim.get("type", "sgd")
        self.type.value = t
        self.lr.value = optim.get("lr", 0.1)
        if t == "quickprop":
            self.mu.value = optim.get("mu", 1.75)
        else:
            self.momentum.value = optim.get("momentum", 0.9)
            self.nesterov.value = optim.get("nesterov", False)


class RegSection:
    def __init__(self) -> None:
        with ui.row().classes("items-end gap-2"):
            self.enabled = ui.checkbox("regolarizzazione")
            self.type = ui.select(list(get_args(RegT)), label="tipo", value="l2").classes("w-24")
            self.lam = ui.number("lambda", value=0.0001, min=0.0, step=0.0001).classes("w-32")
        self.type.bind_visibility_from(self.enabled, "value", backward=lambda v: bool(v))
        self.lam.bind_visibility_from(self.enabled, "value", backward=lambda v: bool(v))

    def to_dict(self) -> Optional[dict]:
        if not self.enabled.value:
            return None
        return {"type": self.type.value, "lam": float(self.lam.value or 0.0)}

    def load(self, reg: Optional[dict]) -> None:
        self.enabled.value = reg is not None
        self.type.value = (reg or {}).get("type", "l2")
        self.lam.value = (reg or {}).get("lam", 0.0001)


class TrainForm:
    """Backs mode=train AND mode=assess (AssessConfig extends TrainConfig,
    schemas.py) — `assess_box` (val_mean/val_std) is only relevant to assess and
    toggles independently of the rest of the form."""

    def __init__(self) -> None:
        with ui.card().classes("w-full") as self.root:
            ui.label("Architettura (layer)").classes("text-bold")
            self.arch = ArchSection()
            ui.separator()
            ui.label("Ottimizzatore").classes("text-bold")
            self.optim = OptimSection()
            ui.separator()
            ui.label("Regolarizzazione (opzionale)").classes("text-bold")
            self.reg = RegSection()
            ui.separator()
            with ui.row().classes("gap-4"):
                self.epochs = ui.number("epochs", value=200, min=1, precision=0).classes("w-28")
                self.batch_size = ui.number("batch_size (vuoto = full batch)", value=None,
                                            min=1, precision=0).classes("w-56")
                self.seed = ui.number("seed", value=0, precision=0).classes("w-24")
                self.n_trials = ui.number("n_trials", value=5, min=1, precision=0).classes("w-24")
            with ui.row().classes("gap-4"):
                self.patience = ui.number("patience (vuoto = disattivo)", value=None,
                                          min=1, precision=0).classes("w-56")
                self.min_delta = ui.number("min_delta", value=0.0, min=0.0, step=0.0001).classes("w-32")
            with ui.row().classes("gap-4 items-end") as self.assess_box:
                ui.label("Solo assess — dal summary di un run select:").classes("text-caption")
                self.val_mean = ui.number("val_mean", value=None).classes("w-32")
                self.val_std = ui.number("val_std", value=None).classes("w-32")

    def build(self, *, assess: bool) -> dict:
        cfg: dict = {
            "arch": self.arch.to_list(),
            "optim": self.optim.to_dict(),
            "epochs": int(self.epochs.value or 1),
            "seed": int(self.seed.value or 0),
            "n_trials": int(self.n_trials.value or 5),
            "min_delta": float(self.min_delta.value or 0.0),
        }
        reg = self.reg.to_dict()
        if reg is not None:
            cfg["reg"] = reg
        if self.batch_size.value not in (None, ""):
            cfg["batch_size"] = int(self.batch_size.value)
        if self.patience.value not in (None, ""):
            cfg["patience"] = int(self.patience.value)
        if assess:
            if self.val_mean.value not in (None, ""):
                cfg["val_mean"] = float(self.val_mean.value)
            if self.val_std.value not in (None, ""):
                cfg["val_std"] = float(self.val_std.value)
        return cfg

    def load(self, cfg: dict) -> None:
        self.arch.load(cfg.get("arch", []))
        self.optim.load(cfg.get("optim", {"type": "sgd", "lr": 0.1}))
        self.reg.load(cfg.get("reg"))
        self.epochs.value = cfg.get("epochs", 200)
        self.batch_size.value = cfg.get("batch_size")
        self.seed.value = cfg.get("seed", 0)
        self.n_trials.value = cfg.get("n_trials", 5)
        self.patience.value = cfg.get("patience")
        self.min_delta.value = cfg.get("min_delta", 0.0)
        self.val_mean.value = cfg.get("val_mean")
        self.val_std.value = cfg.get("val_std")


_DEFAULT_ARCH = [{"units": 3, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                 {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]
_DEFAULT_OPTIM = {"type": "sgd", "lr": 0.1, "momentum": 0.9}
_UNSET = object()


class AxisList:
    """1..N repeatable candidate widgets for one select-grid field. Exactly one
    candidate means a fixed value; two or more means a grid axis to sweep —
    runner/registry.py's `_select` folds a fixed value into a singleton grid
    list internally anyway, so "how many candidates" is the only real
    fixed/grid distinction, which is what lets one widget serve both."""

    def __init__(self, make_row, *, add_label: str = "+ valore", default: Any = None) -> None:
        self._make_row = make_row
        self._default = default
        self.rows: list[Any] = []
        self.container = ui.column().classes("gap-2 w-full")
        ui.button(add_label, icon="add", on_click=lambda: self.add_row()).props("outline dense")
        self.add_row()

    def add_row(self, value: Any = _UNSET) -> None:
        v = self._default if value is _UNSET else value
        self.rows.append(self._make_row(self.container, self._remove, value=v))

    def _remove(self, row: Any) -> None:
        if len(self.rows) <= 1:
            ui.notify("Serve almeno un valore", type="warning")
            return
        row.delete()
        self.rows.remove(row)

    def values(self) -> list:
        return [r.value() for r in self.rows]

    def load_values(self, values: list) -> None:
        for r in list(self.rows):
            r.delete()
        self.rows.clear()
        for v in values:
            self.add_row(value=v)


class ArchCandidate:
    """One `arch` candidate; wraps an ArchSection so a grid axis can hold whole
    layer-list architectures, exactly like TrainForm's own arch field."""

    def __init__(self, container: ui.column, on_remove, *, value: Optional[list[dict]] = None) -> None:
        with container, ui.card().classes("w-full").props("bordered") as self.card:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("architettura").classes("text-caption")
                ui.button(icon="delete", on_click=lambda: on_remove(self)).props("flat dense round")
            self.section = ArchSection()
            if value:
                self.section.load(value)

    def value(self) -> list[dict]:
        return self.section.to_list()

    def delete(self) -> None:
        self.card.delete()


class OptimCandidate:
    """One `optim` candidate; wraps an OptimSection."""

    def __init__(self, container: ui.column, on_remove, *, value: Optional[dict] = None) -> None:
        with container, ui.card().classes("w-full").props("bordered") as self.card:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("ottimizzatore").classes("text-caption")
                ui.button(icon="delete", on_click=lambda: on_remove(self)).props("flat dense round")
            self.section = OptimSection()
            if value:
                self.section.load(value)

    def value(self) -> dict:
        return self.section.to_dict()

    def delete(self) -> None:
        self.card.delete()


class RegCandidate:
    """One `reg` candidate; wraps a RegSection so a candidate can be "off"
    (None) exactly like RegSection's own checkbox does for TrainForm."""

    def __init__(self, container: ui.column, on_remove, *, value: Optional[dict] = None) -> None:
        with container, ui.card().classes("w-full").props("bordered") as self.card:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("regolarizzazione").classes("text-caption")
                ui.button(icon="delete", on_click=lambda: on_remove(self)).props("flat dense round")
            self.section = RegSection()
            if value is not None:
                self.section.load(value)

    def value(self) -> Optional[dict]:
        return self.section.to_dict()

    def delete(self) -> None:
        self.card.delete()


class ScalarCandidate:
    """One numeric candidate for a scalar axis (epochs/batch_size/patience/
    min_delta); an empty field is None, same as TrainForm's "vuoto = ..." fields."""

    def __init__(self, container: ui.column, on_remove, *, value: Optional[float] = None, **kw: Any) -> None:
        with container, ui.row().classes("items-center gap-2") as self.row:
            self.field = ui.number(value=value, **kw).classes("w-32")
            ui.button(icon="delete", on_click=lambda: on_remove(self)).props("flat dense round")

    def value(self) -> Optional[float]:
        return self.field.value if self.field.value not in (None, "") else None

    def delete(self) -> None:
        self.row.delete()


class SelectForm:
    """Backs mode=select (SelectConfig, schemas.py). Reuses TrainForm's own
    building blocks (ArchSection/OptimSection/RegSection) so arch/optim/reg
    look identical here; every field is an AxisList of candidates instead of
    the hand-edited `fixed`/`grid` JSON this replaces (see AxisList's
    docstring for why one widget can serve both fixed values and grid axes)."""

    def __init__(self) -> None:
        with ui.card().classes("w-full") as self.root:
            with ui.row().classes("gap-4"):
                self.k = ui.number("k (fold)", value=5, min=2, precision=0).classes("w-28")
                self.seed = ui.number("seed (vuoto = random)", value=0, precision=0).classes("w-44")
            ui.separator()
            ui.label("Architettura — 1 valore = fissa, più valori = griglia da confrontare").classes("text-bold")
            self.arch = AxisList(ArchCandidate, add_label="+ architettura", default=_DEFAULT_ARCH)
            ui.separator()
            ui.label("Ottimizzatore — 1 valore = fisso, più valori = griglia da confrontare").classes("text-bold")
            self.optim = AxisList(OptimCandidate, add_label="+ ottimizzatore", default=_DEFAULT_OPTIM)
            ui.separator()
            ui.label("Regolarizzazione — 1 valore = fissa, più valori = griglia da confrontare").classes("text-bold")
            self.reg = AxisList(RegCandidate, add_label="+ regolarizzazione", default=None)
            ui.separator()
            ui.label("Epoche / batch / early stopping").classes("text-bold")
            with ui.row().classes("gap-8 items-start"):
                with ui.column().classes("gap-1"):
                    ui.label("epochs").classes("text-caption")
                    self.epochs = AxisList(partial(ScalarCandidate, min=1, precision=0),
                                           add_label="+ epochs", default=80)
                with ui.column().classes("gap-1"):
                    ui.label("batch_size (vuoto = full batch)").classes("text-caption")
                    self.batch_size = AxisList(partial(ScalarCandidate, min=1, precision=0),
                                               add_label="+ batch_size", default=None)
                with ui.column().classes("gap-1"):
                    ui.label("patience (vuoto = disattivo)").classes("text-caption")
                    self.patience = AxisList(partial(ScalarCandidate, min=1, precision=0),
                                             add_label="+ patience", default=None)
                with ui.column().classes("gap-1"):
                    ui.label("min_delta").classes("text-caption")
                    self.min_delta = AxisList(partial(ScalarCandidate, min=0.0, step=0.0001),
                                              add_label="+ min_delta", default=0.0)

    def build(self) -> dict:
        fixed: dict = {}
        grid: dict = {}

        def put(key: str, values: list, *, omit_none_singleton: bool = False) -> None:
            if omit_none_singleton and len(values) == 1 and values[0] is None:
                return
            if len(values) == 1:
                fixed[key] = values[0]
            else:
                grid[key] = values

        put("arch", self.arch.values())
        put("optim", self.optim.values())
        put("reg", self.reg.values(), omit_none_singleton=True)
        put("epochs", [int(v) if v is not None else 1 for v in self.epochs.values()])
        put("batch_size", [int(v) if v is not None else None for v in self.batch_size.values()],
            omit_none_singleton=True)
        put("patience", [int(v) if v is not None else None for v in self.patience.values()],
            omit_none_singleton=True)
        put("min_delta", [float(v) if v is not None else 0.0 for v in self.min_delta.values()])

        cfg: dict = {"k": int(self.k.value or 5), "fixed": fixed, "grid": grid}
        if self.seed.value not in (None, ""):
            cfg["seed"] = int(self.seed.value)
        return cfg

    def load(self, cfg: dict) -> None:
        self.k.value = cfg.get("k", 5)
        self.seed.value = cfg.get("seed")
        fixed, grid = cfg.get("fixed", {}), cfg.get("grid", {})

        def axis_values(key: str, default: Any) -> list:
            if key in grid:
                return list(grid[key])
            if key in fixed:
                return [fixed[key]]
            return [default]

        self.arch.load_values(axis_values("arch", _DEFAULT_ARCH))
        self.optim.load_values(axis_values("optim", _DEFAULT_OPTIM))
        self.reg.load_values(axis_values("reg", None))
        self.epochs.load_values(axis_values("epochs", 80))
        self.batch_size.load_values(axis_values("batch_size", None))
        self.patience.load_values(axis_values("patience", None))
        self.min_delta.load_values(axis_values("min_delta", 0.0))


# --------------------------------------------------------------- results view
def render_summary(summary: dict) -> None:
    """Mode-agnostic on purpose: train/select/assess summaries share no fixed
    key set (runner/registry.py's three handlers each shape their own), so this
    buckets by VALUE SHAPE (scalar / flat list / nested) instead of by key name."""
    scalars, lists, nested = [], {}, {}
    for k, v in summary.items():
        if isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v):
            nested[k] = v
        elif isinstance(v, list):
            lists[k] = v
        elif isinstance(v, dict):
            nested[k] = v
        else:
            scalars.append({"metrica": k, "valore": _fmt_value(v)})

    if scalars:
        ui.table(rows=scalars,
                columns=[{"name": "metrica", "label": "metrica", "field": "metrica"},
                         {"name": "valore", "label": "valore", "field": "valore"}],
                row_key="metrica").classes("w-full")
    for k, v in lists.items():
        ui.label(f"{k}: " + ", ".join(str(_fmt_value(x)) for x in v)).classes("text-caption")
    for k, v in nested.items():
        with ui.expansion(k).classes("w-full"):
            if k == "ranking" and v and isinstance(v[0], dict):
                rows = [{"#": i, **{kk: _fmt_value(vv) for kk, vv in entry.items()
                                    if not isinstance(vv, (dict, list))}}
                       for i, entry in enumerate(v)]
                cols = ["#"] + sorted({kk for r in rows for kk in r} - {"#"})
                ui.table(rows=rows,
                        columns=[{"name": c, "label": c, "field": c, "sortable": True} for c in cols],
                        row_key="#").classes("w-full")
            ui.code(json.dumps(v, indent=2, ensure_ascii=False), language="json").classes("w-full")


async def open_run(task: str, run_id: str) -> None:
    with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
        ui.label(f"{task} / {run_id}").classes("text-h6")
        body = ui.column().classes("w-full")
        with body:
            ui.spinner()
        ui.button("Chiudi", on_click=dialog.close)
    dialog.open()

    try:
        run = await api.get_run(task, run_id)
    except api.ApiError as e:
        body.clear()
        with body:
            ui.label(f"Errore: {e.detail}").classes("text-negative")
        return

    png = await api.get_plot_bytes(task, run_id)
    body.clear()
    with body:
        render_summary(run["summary"])
        if png is not None:
            ui.image(f"data:image/png;base64,{base64.b64encode(png).decode()}").classes("w-full")
        else:
            ui.label("Nessuna curva di apprendimento per questo run "
                     "(tipico di mode=select — una ranking non ha una singola curva).").classes("text-caption")
        with ui.expansion("Config completa").classes("w-full"):
            ui.code(json.dumps(run["config"], indent=2, ensure_ascii=False), language="json").classes("w-full")


# --------------------------------------------------------------------- tabs
def build_submit(on_submitted) -> None:
    task_names = list(TASKS)
    default_task = task_names[0]
    default_modes = list(get_task(default_task).allowed_modes)

    with ui.row().classes("items-end gap-4"):
        task_sel = ui.select(task_names, label="task", value=default_task).classes("w-32")
        mode_sel = ui.select(default_modes, label="mode", value=default_modes[0]).classes("w-32")
        tag_input = ui.input("tag", value="run").classes("w-48")
        template_sel = ui.select({}, label="carica template").classes("w-64")

    train_form = TrainForm()
    select_form = SelectForm()

    with ui.row().classes("items-end gap-2") as assess_prefill_box:
        prefill_run_id = ui.input("run_id di un run select").classes("w-72")
        prefill_btn = ui.button("Precompila da select")

    train_form.root.bind_visibility_from(mode_sel, "value", backward=lambda v: v in ("train", "assess"))
    select_form.root.bind_visibility_from(mode_sel, "value", backward=lambda v: v == "select")
    train_form.assess_box.bind_visibility_from(mode_sel, "value", backward=lambda v: v == "assess")
    assess_prefill_box.bind_visibility_from(mode_sel, "value", backward=lambda v: v == "assess")

    def _sync_templates() -> None:
        matches = [t for t in _TEMPLATES if t.get("mode") == mode_sel.value]
        options = {t["_path"]: f"{t['task']} / {t['tag']} ({t['mode']})" for t in matches}
        template_sel.set_options(options, value=None)

    def _on_task_change() -> None:
        modes = list(get_task(task_sel.value).allowed_modes)
        current = mode_sel.value if mode_sel.value in modes else modes[0]
        mode_sel.set_options(modes, value=current)
        _sync_templates()

    def _apply_template() -> None:
        path = template_sel.value
        if not path:
            return
        tpl = next(t for t in _TEMPLATES if t["_path"] == path)
        tag_input.value = tpl.get("tag", "run")
        cfg = tpl.get("config", {})
        (select_form if tpl["mode"] == "select" else train_form).load(cfg)
        ui.notify(f"Template caricato: {tpl['task']}/{tpl['tag']}", type="info")

    async def _prefill_from_select() -> None:
        run_id = prefill_run_id.value
        if not run_id:
            ui.notify("Inserisci il run_id di un run select", type="warning")
            return
        try:
            run = await api.get_run(task_sel.value, run_id)
        except api.ApiError as e:
            ui.notify(f"Run non trovato: {e.detail}", type="negative")
            return
        summary = run["summary"]
        if "best_config" not in summary:
            ui.notify("Questo run non ha un best_config (non è un run select)", type="warning")
            return
        cfg = dict(summary["best_config"])
        cfg["epochs"] = summary["best_epoch_median"]
        cfg["val_mean"] = summary["val_mean"]
        cfg["val_std"] = summary["val_std"]
        train_form.load(cfg)
        ui.notify("Precompilato dal run select", type="positive")

    def _build_payload() -> dict:
        mode = mode_sel.value
        payload = {"task": task_sel.value, "mode": mode, "tag": tag_input.value or "run"}
        if mode == "select":
            payload["select"] = select_form.build()
        elif mode == "assess":
            payload["assess"] = train_form.build(assess=True)
        else:
            payload["config"] = train_form.build(assess=False)
        return payload

    def _preview() -> None:
        try:
            payload = _build_payload()
        except ValueError as e:
            ui.notify(str(e), type="negative")
            return
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            ui.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json").classes("w-full")
            ui.button("Chiudi", on_click=dialog.close)
        dialog.open()

    async def _submit() -> None:
        try:
            payload = _build_payload()
        except ValueError as e:
            ui.notify(str(e), type="negative")
            return
        try:
            rec = await api.post_job(payload)
        except api.ApiError as e:
            ui.notify(f"Rifiutato ({e.status}): {e.detail}", type="negative", multi_line=True)
            return
        ui.notify(f"Job accodato: {rec['job_id']}", type="positive")
        on_submitted()

    task_sel.on_value_change(_on_task_change)
    mode_sel.on_value_change(_sync_templates)
    template_sel.on_value_change(_apply_template)
    prefill_btn.on_click(_prefill_from_select)

    with ui.row().classes("gap-2 mt-2"):
        ui.button("Anteprima JSON", on_click=_preview).props("outline")
        ui.button("Submit", on_click=_submit).props("color=primary")

    _sync_templates()


def build_jobs() -> None:
    columns = [
        {"name": "job_id", "label": "job_id", "field": "job_id", "sortable": True},
        {"name": "task", "label": "task", "field": "task"},
        {"name": "mode", "label": "mode", "field": "mode"},
        {"name": "status", "label": "status", "field": "status"},
        {"name": "created_fmt", "label": "creato", "field": "created_fmt", "sortable": True},
        {"name": "duration_fmt", "label": "durata", "field": "duration_fmt"},
    ]
    table = ui.table(rows=[], columns=columns, row_key="job_id", selection="single").classes("w-full")

    async def _refresh() -> None:
        try:
            jobs_list = await api.list_jobs()
        except api.ApiError:
            return
        for j in jobs_list:
            j["created_fmt"] = _clock(j["created"])
            j["duration_fmt"] = f"{j['finished'] - j['created']:.1f}s" if j.get("finished") else "…"
        table.rows = jobs_list
        table.update()

    async def _open_selected() -> None:
        if not table.selected:
            ui.notify("Seleziona un job dalla tabella", type="warning")
            return
        row = table.selected[0]
        if row["status"] == "done" and row.get("run_id"):
            await open_run(row["task"], row["run_id"])
        elif row["status"] == "failed":
            with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
                ui.label(f"Job fallito: {row['job_id']}").classes("text-h6 text-negative")
                ui.code(row.get("error") or "(nessun dettaglio)", language=None).classes("w-full")
                ui.button("Chiudi", on_click=dialog.close)
            dialog.open()
        else:
            ui.notify(f"Job ancora in stato: {row['status']}", type="info")

    with ui.row().classes("gap-2"):
        ui.button("Aggiorna ora", on_click=_refresh).props("outline dense")
        ui.button("Apri dettagli / risultati", on_click=_open_selected)
    ui.timer(2.0, _refresh)


def build_runs() -> None:
    task_names = list(TASKS)
    task_sel = ui.select(task_names, label="task", value=task_names[0]).classes("w-40")
    table = ui.table(rows=[], columns=[{"name": "run_id", "label": "run_id", "field": "run_id"}],
                     row_key="run_id", selection="single").classes("w-full")

    async def _refresh() -> None:
        try:
            rows = await api.get_index(task_sel.value)
        except api.ApiError as e:
            ui.notify(f"Errore: {e.detail}", type="negative")
            return
        if rows:
            keys = ["run_id"] + sorted({k for r in rows for k in r} - {"run_id"})
            table.columns = [{"name": k, "label": k, "field": k, "sortable": True} for k in keys]
        table.rows = rows
        table.update()

    async def _open_selected() -> None:
        if not table.selected:
            ui.notify("Seleziona un run dalla tabella", type="warning")
            return
        await open_run(task_sel.value, table.selected[0]["run_id"])

    with ui.row().classes("gap-2"):
        ui.button("Carica run", on_click=_refresh).props("outline dense")
        ui.button("Apri run", on_click=_open_selected)
    ui.timer(0.5, _refresh, once=True)


# --------------------------------------------------------------------- page
@ui.page("/")
def console() -> None:
    ui.label("ml_proj — console esperimenti").classes("text-h5")
    with ui.tabs().classes("w-full") as tabs:
        submit_tab = ui.tab("Submit")
        jobs_tab = ui.tab("Jobs")
        runs_tab = ui.tab("Runs")
    with ui.tab_panels(tabs, value=submit_tab).classes("w-full"):
        with ui.tab_panel(submit_tab):
            build_submit(on_submitted=lambda: tabs.set_value(jobs_tab))
        with ui.tab_panel(jobs_tab):
            build_jobs()
        with ui.tab_panel(runs_tab):
            build_runs()


ui.run_with(app, mount_path="/gui", title="ml_proj — console esperimenti", dark=None)
