"""Request validation for the service API"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

Act = Literal["identity", "sigmoid", "tanh", "relu"]
Init = Literal["uniform", "glorot", "he"]
RegT = Literal["l2", "l1"]
Task = Literal["monk1", "monk2", "monk3", "cup"]
Mode = Literal["train", "select", "assess"]


class LayerSpec(BaseModel):
    units: int = Field(gt=0)
    act: Act
    init: Init = "uniform"
    init_kwargs: dict[str, Any] = Field(default_factory=dict)


class SgdSpec(BaseModel):
    type: Literal["sgd"] = "sgd"
    lr: float = Field(gt=0)
    momentum: float = Field(0.0, ge=0, le=1)
    nesterov: bool = False


class QuickPropSpec(BaseModel):
    type: Literal["quickprop"] = "quickprop"
    lr: float = Field(0.1, gt=0)
    mu: float = Field(1.75, gt=0)


OptimSpec = Annotated[Union[SgdSpec, QuickPropSpec], Field(discriminator="type")]


class RegSpec(BaseModel):
    type: RegT
    lam: float = Field(ge=0)


class TrainConfig(BaseModel):
    """Payload for mode=train and mode=assess (assess adds val_mean/val_std)."""

    arch: list[LayerSpec] = Field(min_length=1)
    loss: Literal["mse"] = "mse"
    optim: OptimSpec
    reg: Optional[RegSpec] = None
    epochs: int = Field(gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    seed: int = 0
    n_trials: int = Field(5, gt=0)
    # early stopping
    patience: Optional[int] = Field(default=None, gt=0)
    min_delta: float = Field(0.0, ge=0)


class AssessConfig(TrainConfig):
    """TrainConfig + validation figures from a prior `select` run"""

    val_mean: Optional[float] = None
    val_std: Optional[float] = None


class SelectConfig(BaseModel):
    """Payload for mode=select

    `grid` axis values are intentionally left as `Any`: an axis can hold full
    layer-list architectures, optimizer dicts, regularizer dicts, or plain
    scalars (e.g. sweeping epochs itself) — iter_grid's cartesian product
    doesn't care, and forcing a rigid per-axis schema here would fight that
    flexibility for little gain
    """

    k: int = Field(5, gt=0)
    seed: Optional[int] = None
    fixed: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)


class JobRequest(BaseModel):
    task: Task
    mode: Mode = "train"
    tag: str = "run"
    config: Optional[TrainConfig] = None
    select: Optional[SelectConfig] = None
    assess: Optional[AssessConfig] = None

    @model_validator(mode="after")
    def _matching_field_is_set(self) -> "JobRequest":
        needed = {"train": "config", "select": "select", "assess": "assess"}[self.mode]
        if getattr(self, needed) is None:
            raise ValueError(f"mode={self.mode!r} requires the {needed!r} field to be set")
        return self

    def payload(self) -> dict:
        """The plain dict runner.engine.run_experiment expects as `payload`."""
        model = {"train": self.config, "select": self.select, "assess": self.assess}[self.mode]
        return model.model_dump(exclude_none=True)
