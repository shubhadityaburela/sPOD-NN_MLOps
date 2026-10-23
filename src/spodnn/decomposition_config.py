"""Decomposition configs: YAML files in configs/decomposition, validated into typed, read-only objects.

Each case (synthetic, wildfire 1D, 2D, 2D_ww) has one file holding every
number that defines its sPOD decomposition; configs/decomposition/<case>.yaml
decomposes data/<case>.zarr. Loading checks types, ranges and unknown keys, so
a typo fails at load time instead of producing a silently wrong decomposition.

The settings are kept apart from the data configs (spodnn.config) on purpose:
the data's fingerprint and its DVC stage cover the whole data config and
config.py, so a decomposition setting there would regenerate the data for nothing.
Checks that need the dataset (kind matches, scipy transforms only in 2D,
shift_modes only for 2D_ww) happen when the decomposition runs.
"""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, PositiveFloat, PositiveInt, StrictBool, TypeAdapter, model_validator

from spodnn.config import BaseConfig


class TransformConfig(BaseConfig):
    """How the sPOD transforms shift a field (arguments of spod.transforms).

    use_scipy_transform: shift with scipy.ndimage (cubic splines) instead of
        sPOD's own Lagrange interpolation matrices; the paper uses it for 2D only.
    interp_order: order of sPOD's own 1D interpolation (1, 3 or 5). Ignored by
        the scipy shift and by the general 2D shift (both use a fixed cubic stencil).
    """

    use_scipy_transform: StrictBool
    interp_order: PositiveInt

    @model_validator(mode="after")
    def validate_interp_order(self) -> Self:
        if self.interp_order not in (1, 3, 5):
            raise ValueError("The interpolation order has to be amongst [1, 3, 5]")
        return self


class sPODConfig(BaseConfig):
    """Settings of one shifted_rPCA run (one variable).

    n_iter: maximum number of iterations.
    mu_factor, lambda_factor: scale the penalty parameters mu0 and lambda0
        (formulas: see `penalty` in the case configs).
    nmodes: modes kept per frame; None keeps the rank sPOD settles on.
    nmodes_max: largest rank sPOD may use per frame (at least nmodes).
    eps: stop once the relative residual falls below eps.
    """

    n_iter: PositiveInt
    mu_factor: PositiveFloat
    lambda_factor: PositiveFloat
    nmodes: PositiveInt | None = None
    nmodes_max: PositiveInt
    eps: PositiveFloat

    @model_validator(mode="after")
    def validate_mode_val(self) -> Self:
        if self.nmodes is not None and self.nmodes > self.nmodes_max:
            raise ValueError("Required nmodes <= nmodes_max")
        return self


class SyntheticDecompositionConfig(BaseConfig):
    """Decomposition of the synthetic case: one field, two frames, no shift POD.

    penalty: formula for the sPOD penalty parameters of a snapshot matrix q of
        shape (N, M), see WildfireDecompositionConfig.
    """

    kind: Literal["synthetic"] = "synthetic"
    penalty: Literal["matrix", "grid"] = "matrix"
    dtype: Literal["float32", "float64"] = "float32"  # Restricts values to these strings
    shift_modes: None = None

    transform: TransformConfig
    spod: sPODConfig


class WildfireSPODConfig(BaseConfig):
    """One sPOD run per variable: temperature T and supply mass fraction S."""

    T: sPODConfig
    S: sPODConfig


class WildfireDecompositionConfig(BaseConfig):
    """Decomposition of a wildfire case (1D, 2D or 2D_ww), T and S separately.

    penalty: which of the paper's two formulas sets mu0 and lambda0, for a
        snapshot matrix q of shape (N, M) on an Nx x Ny grid:
        "matrix" (1D, synthetic): mu0 = N M / (4 sum|q|) mu_factor,
                                  lambda0 = lambda_factor / sqrt(max(N, M));
        "grid" (2D, 2D_ww):       mu0 = N / (4 sum|q|) mu_factor,
                                  lambda0 = lambda_factor / sqrt(max(Nx, Ny)).
    shift_modes: rank of the POD of the radial shifts (2D_ww: 3); None where
        the shifts are used as they are.
    """

    kind: Literal["wildfire"] = "wildfire"
    penalty: Literal["matrix", "grid"]
    dtype: Literal["float32", "float64"] = "float32"  # Restricts values to these strings
    shift_modes: PositiveInt | None = None

    transform: TransformConfig
    spod: WildfireSPODConfig


DecompositionConfig = Annotated[
    SyntheticDecompositionConfig | WildfireDecompositionConfig,
    Field(discriminator="kind"),
]


def load_decomposition_config(path: str | Path) -> DecompositionConfig:
    """Loads a YAML file and validates it into a DecompositionConfig instance."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _case_adapter = TypeAdapter(DecompositionConfig)
    return _case_adapter.validate_python(data)
