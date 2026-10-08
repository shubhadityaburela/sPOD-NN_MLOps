from spod.transforms import transforms
from scipy.special import eval_hermite
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Grid:
    """A class to represent a grid for synthetic data generation."""
    Nx: int = 500
    Ny: int = 1
    Nt: int = 500
    T: float = 1.0
    L: float = 1.0

    @property
    def x(self) -> np.ndarray:
        return np.arange(-self.Nx // 2, self.Nx // 2) / self.Nx * self.L

    @property
    def t(self) -> np.ndarray:
        return np.arange(0, self.Nt) / self.Nt * self.T

    @property
    def Xgrid(self) -> np.ndarray:
        return np.meshgrid(self.x, self.t, indexing='ij')[0]

    @property
    def Tgrid(self) -> np.ndarray:
        return np.meshgrid(self.x, self.t, indexing='ij')[1]

    @property
    def dx(self) -> float:
        return self.x[1] - self.x[0]

    @property
    def dt(self) -> float:
        return self.t[1] - self.t[0]


@dataclass
class SyntheticData:
    q: np.ndarray            # (Nx, Nt * n_samples)
    frames: list             # [q1, q2], the co-moving frame fields before shifting
    shifts: list             # [shifts_frame1, shifts_frame2], each (1, Nt * n_samples)
    params: np.ndarray       # (2, Nt * n_samples), rows [t, mu]
    transforms: list         # the two spod transforms


def generate(mu_values: np.ndarray, grid: Grid(), n_terms: int=8) -> SyntheticData:

    # gauss hermite polynomials of order n
    psi = lambda n, x: np.exp(-x ** 2 / 2) * eval_hermite(n, x)
    Nsamples = np.size(mu_values, 0)
    w = 0.015 * grid.L
    Nt = np.size(grid.Tgrid, 1)

    qs1 = []
    qs2 = []
    for k in range(Nsamples):  # loop over all possible mu vectors
        q1 = np.zeros_like(grid.Xgrid)
        q2 = np.zeros_like(grid.Xgrid)
        for n in range(n_terms):  # loop over all components of the vector
            q1 += (1 + np.exp(-2 * n * grid.Tgrid)) * mu_values[k] * np.cos(
                - 2 * np.pi * grid.Tgrid / grid.T * (n + 1)) * psi(n, (
                    grid.Xgrid + 0.1 * grid.L) / w)
            q2 += (1 + np.exp(-2 * n * grid.Tgrid)) * mu_values[k] * np.cos(
                - 2 * np.pi * grid.Tgrid / grid.T * (n + 1)) * psi(n, (
                    grid.Xgrid - 0.1 * grid.L) / w)

        qs1.append(q1)
        qs2.append(-q2)

    q1 = np.concatenate(qs1, axis=1)
    q2 = np.concatenate(qs2, axis=1)
    q_frames = [q1, q2]

    shifts = [np.asarray([mu_values[n] * grid.t]) for n in range(Nsamples)]

    shifts = [np.concatenate(shifts, axis=1), -np.concatenate(shifts, axis=1)]
    data_shape = [grid.Nx, 1, 1, Nt * Nsamples]
    trafos = [
        transforms(data_shape, [grid.L], shifts=np.squeeze(shifts[0]), dx=[grid.dx], use_scipy_transform=False,
                    interp_order=5),
        transforms(data_shape, [grid.L], shifts=np.squeeze(shifts[1]), dx=[grid.dx], use_scipy_transform=False,
                    interp_order=5)]

    q = 0
    for trafo, qf in zip(trafos, q_frames):
        q += trafo.apply(qf)

    # Parameter matrix
    p = [np.squeeze(np.asarray([[grid.t], [np.ones_like(grid.t) * mu]])) for mu in mu_values]
    p = np.concatenate(p, axis=1)

    return SyntheticData(q, q_frames, shifts, p, trafos)