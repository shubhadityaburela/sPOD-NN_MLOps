"""Case configs: YAML files in configs/, validated into typed, read-only objects.

Each case (synthetic, wildfire 1D, 2D, 2D_ww) has one file holding every
number that defines its data. Loading checks types, ranges and unknown keys,
so a typo fails at load time instead of producing silently wrong data.
"""

import hashlib
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, StrictBool, TypeAdapter, model_validator


class BaseConfig(BaseModel):
    """Base for all config sections: unknown keys are errors, values are read-only."""

    model_config = ConfigDict(
        extra="forbid",  # Rejects unknown keys (catches typos like `cfI: 0.8`)
        frozen=True,  # Makes config objects read-only so settings can't change mid-run
    )


class SyntheticGridConfig(BaseConfig):
    """Space-time grid of the synthetic case (see spodnn.synthetic.Grid)."""

    L: PositiveFloat
    T: PositiveFloat
    Nx: PositiveInt
    Nt: PositiveInt


class SplitConfig(BaseConfig):
    """Parameter values (mu or beta) for training and testing.

    Training values must be strictly increasing (the interpolation in mu
    requires it) and every test value must lie inside the training range.
    """

    train: list[float]
    test: list[float]

    @model_validator(mode="after")
    def validate_mu_splits(self) -> Self:

        if self.train == [] or self.test == []:
            raise ValueError("training and testing sets should have values")

        if self.train != sorted(set(self.train)):
            raise ValueError(f"train values must be strictly increasing, got {self.train}")

        min_train, max_train = min(self.train), max(self.train)
        for mu in self.test:
            if not (min_train <= mu <= max_train):
                raise ValueError(f"Test value mu={mu} lies outside the training range [{min_train}, {max_train}]")
        return self


class SyntheticCaseConfig(BaseConfig):
    """The synthetic case: analytic snapshots on a grid, one set per mu."""

    kind: Literal["synthetic"] = "synthetic"
    n_terms: PositiveInt
    dtype: Literal["float32", "float64"] = "float32"  # Restricts values to these strings
    grid: SyntheticGridConfig
    mu: SplitConfig


class WildfireGridConfig(BaseConfig):
    """Domain size, grid points and solver time steps; Ny == 1 is the 1D case."""

    Lx: PositiveFloat
    Ly: PositiveFloat
    Nx: PositiveInt
    Ny: PositiveInt
    Nt: PositiveInt


class WildfirePhysicsConfig(BaseConfig):
    """Wind speeds, CFL number and speed of sound; v_y must be 0 (the solver ignores it)."""

    v_x: float
    v_y: float
    cfl: float = Field(gt=0, le=1)
    c: PositiveFloat
    ThermalDiffusivity: PositiveFloat
    PreExponentialFactor: PositiveFloat
    TempRisePerSecond: PositiveFloat
    ScaledHeatTransferCoeff: PositiveFloat
    TAmbient: float = Field(gt=273.15)

    @model_validator(mode="after")
    def validate_v_y(self) -> Self:
        if self.v_y != 0:
            raise ValueError("v_y is ignored by the solver, therefore must be set 0")
        return self


class WildfireCaseConfig(BaseConfig):
    """A wildfire case: solver settings, every `step`-th time step saved, one run per beta.

    Nt must be divisible by step, and edge_detect (2D shift detection) needs Ny > 1.
    """

    kind: Literal["wildfire"] = "wildfire"
    step: PositiveInt
    dtype: Literal["float32", "float64"] = "float32"  # Restricts values to these strings
    edge_detect: StrictBool
    grid: WildfireGridConfig
    physics: WildfirePhysicsConfig
    beta: SplitConfig

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        # Rule 1: Nt must be evenly divisible by step
        if self.grid.Nt % self.step != 0:
            raise ValueError(f"Nt ({self.grid.Nt}) must be divisible by step ({self.step}).")

        return self

    @model_validator(mode="after")
    def validate_edge_detect(self) -> Self:
        if self.edge_detect and self.grid.Ny == 1:
            raise ValueError("edge_detect should be false if the problem is 1D")

        return self

    @model_validator(mode="after")
    def case_edge_detect(self) -> Self:
        if self.edge_detect:
            if self.grid.Lx == self.grid.Nx and self.grid.Ly == self.grid.Ny:
                return self
            else:
                raise ValueError("edge_detect will only work if Nx = Lx and Ny = Ly")
        return self


CaseConfig = Annotated[
    SyntheticCaseConfig | WildfireCaseConfig,
    Field(discriminator="kind"),
]


def load_config(path: str | Path) -> CaseConfig:
    """Loads a YAML file and validates it into a CaseConfig instance."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _case_adapter = TypeAdapter(CaseConfig)
    return _case_adapter.validate_python(data)


def config_hash(cfg: CaseConfig) -> str:
    """
    Computes a SHA-256 hash from canonical JSON representation.
    Comments, white space, or field reordering in YAML won't alter the hash.
    """
    canonical_json = cfg.model_dump_json(by_alias=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
