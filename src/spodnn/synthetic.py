"""Synthetic test case: two structures travelling in opposite directions.

The field is the sum of two co-moving frames, each shifted in space by its own
parameter-dependent shift:

    q(x, t; mu) = T_{Delta_1} q1(x, t; mu) + T_{Delta_2} q2(x, t; mu),
    Delta_1(t; mu) = mu t,   Delta_2(t; mu) = -mu t,

where T_Delta shifts a field by Delta in x (periodic, 5th-order interpolation,
spod.transforms). In their own frames the two structures are

    q1(x, t; mu) =  mu * sum_{n=0}^{N-1} (1 + exp(-2 n t)) cos(2 pi (n + 1) t / T) psi_n((x + 0.1 L) / w)
    q2(x, t; mu) = -mu * sum_{n=0}^{N-1} (1 + exp(-2 n t)) cos(2 pi (n + 1) t / T) psi_n((x - 0.1 L) / w)

with Gauss-Hermite functions psi_n(xi) = exp(-xi^2 / 2) H_n(xi) (physicists'
Hermite polynomials H_n), width w = 0.015 L and N = n_terms terms.

Ported from paper/synthetic_sup.py (create_data and the grid set-up in __init__);
the numbers match the paper code to about 1e-14 (tests/test_synthetic.py).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_hermite
from spod.transforms import transforms


@dataclass(frozen=True)
class Grid:
    """Uniform space-time grid: Nx points in x on [-L/2, L/2), Nt points in t on [0, T).

    Derived quantities are properties, computed from the fields on access.
    """

    Nx: int = 500
    Ny: int = 1
    Nt: int = 500
    T: float = 1.0
    L: float = 1.0

    @property
    def x(self) -> np.ndarray:
        """Spatial points x_i = (i - Nx/2) L / Nx, i = 0..Nx-1."""
        return np.arange(-self.Nx // 2, self.Nx // 2) / self.Nx * self.L

    @property
    def t(self) -> np.ndarray:
        """Time points t_j = j T / Nt, j = 0..Nt-1."""
        return np.arange(0, self.Nt) / self.Nt * self.T

    @property
    def Xgrid(self) -> np.ndarray:
        """x on the space-time grid, shape (Nx, Nt): Xgrid[i, j] = x_i."""
        return np.meshgrid(self.x, self.t, indexing="ij")[0]

    @property
    def Tgrid(self) -> np.ndarray:
        """t on the space-time grid, shape (Nx, Nt): Tgrid[i, j] = t_j."""
        return np.meshgrid(self.x, self.t, indexing="ij")[1]

    @property
    def dx(self) -> float:
        """Spatial step L / Nx."""
        return self.x[1] - self.x[0]

    @property
    def dt(self) -> float:
        """Time step T / Nt."""
        return self.t[1] - self.t[0]


@dataclass
class SyntheticData:
    """Snapshots of the synthetic case for one or more parameter values.

    Samples are stacked along the time axis: columns k*Nt .. (k+1)*Nt - 1 belong
    to the k-th value of mu.
    """

    q: np.ndarray  # (Nx, Nt * n_samples)
    frames: list  # [q1, q2], the co-moving frame fields before shifting
    shifts: list  # [shifts_frame1, shifts_frame2], each (1, Nt * n_samples)
    params: np.ndarray  # (2, Nt * n_samples), rows [t, mu]
    transforms: list  # the two spod transforms


def generate(mu_values: np.ndarray, grid: Grid, n_terms: int = 8) -> SyntheticData:
    """Generate the synthetic snapshots q(x, t; mu) for each mu in mu_values.

    Args:
        mu_values: Parameter values mu, one sample per value. They set both the
            amplitude of the structures and their speed (shift mu t).
        grid: Space-time grid.
        n_terms: Number N of Gauss-Hermite terms per structure. The paper uses
            N = 8, the same number as the sPOD modes it keeps per frame; the two
            are independent here.

    Returns:
        The snapshots q, the unshifted frames [q1, q2], the shifts
        [mu t, -mu t] (each of shape (1, Nt * n_samples)), the network inputs
        with rows [t, mu] (the order the committed networks were trained on), and
        the two transforms that apply the shifts.
    """

    # gauss hermite polynomials of order n
    psi = lambda n, x: np.exp(-(x**2) / 2) * eval_hermite(n, x)  # noqa: E731 (lambda is fine here)
    Nsamples = np.size(mu_values, 0)
    w = 0.015 * grid.L
    Nt = np.size(grid.Tgrid, 1)

    XX, TT = grid.Xgrid, grid.Tgrid

    qs1 = []
    qs2 = []
    for k in range(Nsamples):  # loop over all possible mu vectors
        q1 = np.zeros_like(XX)
        q2 = np.zeros_like(XX)
        for n in range(n_terms):  # loop over all components of the vector
            q1 += (
                (1 + np.exp(-2 * n * TT))
                * mu_values[k]
                * np.cos(-2 * np.pi * TT / grid.T * (n + 1))
                * psi(n, (XX + 0.1 * grid.L) / w)
            )
            q2 += (
                (1 + np.exp(-2 * n * TT))
                * mu_values[k]
                * np.cos(-2 * np.pi * TT / grid.T * (n + 1))
                * psi(n, (XX - 0.1 * grid.L) / w)
            )

        qs1.append(q1)
        qs2.append(-q2)

    q1 = np.concatenate(qs1, axis=1)
    q2 = np.concatenate(qs2, axis=1)
    q_frames = [q1, q2]

    shifts = [np.asarray([mu_values[n] * grid.t]) for n in range(Nsamples)]

    shifts = [np.concatenate(shifts, axis=1), -np.concatenate(shifts, axis=1)]
    data_shape = [grid.Nx, 1, 1, Nt * Nsamples]
    trafos = [
        transforms(
            data_shape, [grid.L], shifts=np.squeeze(shifts[0]), dx=[grid.dx], use_scipy_transform=False, interp_order=5
        ),
        transforms(
            data_shape, [grid.L], shifts=np.squeeze(shifts[1]), dx=[grid.dx], use_scipy_transform=False, interp_order=5
        ),
    ]

    q = 0
    for trafo, qf in zip(trafos, q_frames, strict=True):
        q += trafo.apply(qf)

    # Parameter matrix
    p = [np.squeeze(np.asarray([[grid.t], [np.ones_like(grid.t) * mu]])) for mu in mu_values]
    p = np.concatenate(p, axis=1)

    return SyntheticData(q, q_frames, shifts, p, trafos)
