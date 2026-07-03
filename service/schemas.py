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


class LinearDecaySpec(BaseModel):

    type: Literal["linear_decay"] = "linear_decay"
    eta_0: float = Field(gt=0)
    tau: int = Field(gt=0)
    eta_tau: Optional[float] = Field(default=None, ge=0)


# lr is either a plain positive number or a schedule dict — the numeric branch
# carries its own `gt=0` since Pydantic can't apply a bare constraint across a Union.
LrSpec = Union[Annotated[float, Field(gt=0)], LinearDecaySpec]


class SgdSpec(BaseModel):
    type: Literal["sgd"] = "sgd"
    lr: LrSpec
    momentum: float = Field(0.0, ge=0, le=1)
    nesterov: bool = False


class QuickPropSpec(BaseModel):
    type: Literal["quickprop"] = "quickprop"
    lr: float = Field(0.1, gt=0)
    mu: float = Field(1.75, gt=0)


class AdaGradSpec(BaseModel):
    type: Literal["adagrad"] = "adagrad"
    lr: LrSpec = 0.01
    epsilon: float = Field(1e-8, gt=0)


OptimSpec = Annotated[Union[SgdSpec, QuickPropSpec, AdaGradSpec], Field(discriminator="type")]


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

    """

    k: int = Field(5, gt=0)
    seed: Optional[int] = None
    n_core: Optional[int] = -1
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
