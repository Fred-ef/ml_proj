"""Request validation for the service API."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from src.model_selection.grid_search import expand_grid_value

Act = Literal["identity", "sigmoid", "tanh", "relu"]
Init = Literal["uniform", "glorot", "he"]
RegT = Literal["l2", "l1"]
Task = Literal["monk1", "monk2", "monk3", "cup"]
Mode = Literal["train", "select", "assess"]
Loss = Literal["mse", "cross_entropy"]


class StrictModel(BaseModel):
    """Reject misspelled or unsupported fields instead of silently dropping them."""

    model_config = ConfigDict(extra="forbid")


class LayerSpec(StrictModel):
    units: int = Field(gt=0)
    act: Act
    init: Init = "uniform"
    init_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _initializer_arguments_match(self) -> "LayerSpec":
        if self.init == "uniform":
            unknown = set(self.init_kwargs) - {"scale"}
            if unknown:
                raise ValueError(f"unsupported uniform initializer arguments: {sorted(unknown)}")
            scale = self.init_kwargs.get("scale", 0.1)
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError("uniform initializer scale must be positive")
        elif self.init_kwargs:
            raise ValueError(f"initializer {self.init!r} does not accept init_kwargs")
        return self


class LinearDecaySpec(StrictModel):
    type: Literal["linear_decay"] = "linear_decay"
    eta_0: float = Field(gt=0)
    tau: int = Field(gt=0)
    eta_tau: Optional[float] = Field(default=None, ge=0)


LrSpec = Union[Annotated[float, Field(gt=0)], LinearDecaySpec]


class SgdSpec(StrictModel):
    type: Literal["sgd"] = "sgd"
    lr: LrSpec
    momentum: float = Field(0.0, ge=0, le=1)
    nesterov: bool = False


class QuickPropSpec(StrictModel):
    type: Literal["quickprop"] = "quickprop"
    lr: float = Field(0.1, gt=0)
    mu: float = Field(1.75, gt=0)


class AdaGradSpec(StrictModel):
    type: Literal["adagrad"] = "adagrad"
    lr: LrSpec = 0.01
    epsilon: float = Field(1e-8, gt=0)


OptimSpec = Annotated[Union[SgdSpec, QuickPropSpec, AdaGradSpec], Field(discriminator="type")]


class RegSpec(StrictModel):
    type: RegT
    lam: float = Field(ge=0)


class TrainConfig(StrictModel):
    """Payload for a plain training run (assessment adds validation figures)."""

    arch: list[LayerSpec] = Field(min_length=1)
    loss: Loss = "mse"
    optim: OptimSpec
    reg: Optional[RegSpec] = None
    epochs: int = Field(gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    scale: bool = False
    seed: int = 0
    n_trials: int = Field(5, gt=0)

    @model_validator(mode="after")
    def _optimizer_matches_batching(self) -> "TrainConfig":
        if isinstance(self.optim, QuickPropSpec) and self.batch_size is not None:
            raise ValueError("QuickProp requires full-batch training (batch_size must be null)")
        return self


class AssessConfig(TrainConfig):
    """Fixed-epoch training on the development set plus prior CV estimates."""

    val_mean: Optional[float] = None
    val_std: Optional[float] = None


_GRID_FIELDS = {
    "arch", "loss", "optim", "reg", "epochs", "batch_size", "scale",
    "patience", "min_delta", "monitor",
}
_LOSS_ADAPTER = TypeAdapter(Loss)
_ARCH_ADAPTER = TypeAdapter(list[LayerSpec])
_OPTIM_ADAPTER = TypeAdapter(OptimSpec)
_REG_ADAPTER = TypeAdapter(Optional[RegSpec])


def _validate_grid_value(key: str, value: Any) -> None:
    """Validate one already-unwrapped fixed/grid candidate."""

    if key == "arch":
        _ARCH_ADAPTER.validate_python(value)
    elif key == "loss":
        _LOSS_ADAPTER.validate_python(value)
    elif key == "optim":
        _OPTIM_ADAPTER.validate_python(value)
    elif key == "reg":
        _REG_ADAPTER.validate_python(value)
    elif key in {"epochs", "batch_size", "patience"}:
        if value is None and key in {"batch_size", "patience"}:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    elif key == "min_delta":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("min_delta must be a non-negative number")
    elif key == "scale":
        if not isinstance(value, bool):
            raise ValueError("scale must be boolean")
    elif key == "monitor":
        if value not in {"val_loss", "val_acc", "val_mee", "val_mse"}:
            raise ValueError("unsupported early-stopping monitor")


class SelectConfig(StrictModel):
    """K-fold or single-holdout hyperparameter-grid payload."""

    k: int = Field(5, ge=2)
    validation_fraction: Optional[float] = Field(default=None, gt=0, lt=1)
    seed: Optional[Union[int, Literal["auto"]]] = None
    split_seed: Optional[int] = None
    stratify: Optional[bool] = None
    n_core: Optional[int] = -1
    fixed: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_search_space(self) -> "SelectConfig":
        if self.validation_fraction is not None and "k" in self.model_fields_set:
            raise ValueError("choose either k-fold CV or validation_fraction holdout")
        if self.n_core == 0 or (self.n_core is not None and self.n_core < -2):
            raise ValueError("n_core must be null, -2, -1, or a positive integer")
        keys = set(self.fixed) | set(self.grid)
        unknown = keys - _GRID_FIELDS
        if unknown:
            raise ValueError(f"unsupported selection fields: {sorted(unknown)}")
        overlap = set(self.fixed) & set(self.grid)
        if overlap:
            raise ValueError(f"fields cannot be both fixed and in grid: {sorted(overlap)}")
        missing = {"arch", "optim", "epochs"} - keys
        if missing:
            raise ValueError(f"selection requires fields: {sorted(missing)}")
        for key, values in self.grid.items():
            if not values:
                raise ValueError(f"grid axis {key!r} cannot be empty")
            for value in values:
                for expanded in expand_grid_value(value):
                    _validate_grid_value(key, expanded)
        for key, value in self.fixed.items():
            _validate_grid_value(key, value)

        optim_values = self.grid.get("optim", [self.fixed.get("optim")])
        batch_values = self.grid.get("batch_size", [self.fixed.get("batch_size")])
        if any((value or {}).get("type") == "quickprop" for value in optim_values):
            if any(value is not None for value in batch_values):
                raise ValueError("QuickProp requires full-batch training (batch_size must be null)")
        return self


class JobRequest(StrictModel):
    task: Task
    mode: Mode = "train"
    tag: str = "run"
    config: Optional[TrainConfig] = None
    select: Optional[SelectConfig] = None
    assess: Optional[AssessConfig] = None

    @model_validator(mode="after")
    def _matching_field_is_set(self) -> "JobRequest":
        needed = {"train": "config", "select": "select", "assess": "assess"}[self.mode]
        selected = getattr(self, needed)
        if selected is None:
            raise ValueError(f"mode={self.mode!r} requires the {needed!r} field to be set")
        extras = [name for name in ("config", "select", "assess")
                  if name != needed and getattr(self, name) is not None]
        if extras:
            raise ValueError(f"mode={self.mode!r} does not accept fields: {extras}")

        expected_outputs = 4 if self.task == "cup" else 1
        if self.mode == "select":
            arches = selected.grid.get("arch", [selected.fixed.get("arch")])
        else:
            arches = [selected.arch]
        for arch in arches:
            last_units = arch[-1].units if isinstance(arch[-1], LayerSpec) else arch[-1]["units"]
            if last_units != expected_outputs:
                raise ValueError(
                    f"task={self.task!r} requires {expected_outputs} output unit(s), got {last_units}"
                )
        if self.mode == "select":
            losses = selected.grid.get("loss", [selected.fixed.get("loss", "mse")])
        else:
            losses = [selected.loss]
        if "cross_entropy" in losses:
            if self.task == "cup":
                raise ValueError("cross_entropy is only supported for binary classification tasks")
            for arch in arches:
                last_act = arch[-1].act if isinstance(arch[-1], LayerSpec) else arch[-1]["act"]
                if last_act != "sigmoid":
                    raise ValueError("cross_entropy requires a sigmoid output layer")
        return self

    def payload(self) -> dict:
        """The plain dict runner.engine.run_experiment expects as ``payload``."""

        model = {"train": self.config, "select": self.select, "assess": self.assess}[self.mode]
        payload = model.model_dump(exclude_none=True)
        if self.mode == "select" and self.select.validation_fraction is not None:
            payload.pop("k", None)  # Pydantic includes the unused CV default.
        return payload
