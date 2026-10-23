"""From a case config to arrays: the physics half of `spodnn generate`.

generate_case yields one Sample per parameter value (train, then test) and
writes nothing; spodnn.dataset stores the samples.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from spodnn.config import SyntheticCaseConfig, WildfireCaseConfig
from spodnn.synthetic import Grid, generate
from spodnn.wildfire.shifts import shifts_1d, shifts_2d, shifts_2d_edges
from spodnn.wildfire.solver import solve


@dataclass(frozen=True)
class Sample:
    """The data for one parameter value."""

    param: float  # β or μ
    q: np.ndarray  # snapshots (rows, Nt), float64 straight from the solver
    shifts: np.ndarray  # 1D (3, Nt) · 2D (2, 2, Nt) · 2D_ww compact (N_θ, Nt) · synthetic (2, Nt)
    x: np.ndarray
    y: np.ndarray  # zeros(1) for 1D and synthetic
    t: np.ndarray
    frames: np.ndarray | None = None  # synthetic only: (2, Nx, Nt)


def generate_case(cfg) -> Iterator[Sample]:
    """Solve the case for each parameter value in turn, yielding each Sample as soon as it is ready.

    A generator, so only one parameter value's arrays are in memory at a time.
    """
    if isinstance(cfg, WildfireCaseConfig):
        yield from _wildfire(cfg)
    else:
        yield from _synthetic(cfg)


def _wildfire(cfg: WildfireCaseConfig) -> Iterator[Sample]:
    """Solver run per beta, with the shifts for its geometry: 1D, circular 2D, or any 2D front (edges)."""
    for beta in cfg.beta.train + cfg.beta.test:
        sol = solve(cfg.grid, cfg.physics, beta, cfg.step)
        if cfg.grid.Ny == 1:
            shifts = shifts_1d(sol.q, sol.x)
        else:
            if cfg.edge_detect:
                shifts = shifts_2d_edges(sol.q, sol.x, sol.y)
            else:
                shifts = shifts_2d(sol.q, sol.x, sol.y)
        yield Sample(param=beta, q=sol.q, shifts=shifts, x=sol.x, y=sol.y, t=sol.t)


def _synthetic(cfg: SyntheticCaseConfig) -> Iterator[Sample]:
    """Analytic snapshots per mu, with the two co-moving frames and their shifts."""
    grid = Grid(Nx=cfg.grid.Nx, Ny=1, Nt=cfg.grid.Nt, T=cfg.grid.T, L=cfg.grid.L)
    for mu in cfg.mu.train + cfg.mu.test:
        data = generate([mu], grid, cfg.n_terms)
        yield Sample(
            param=mu,
            q=data.q,
            shifts=np.concatenate(data.shifts),
            x=grid.x,
            y=np.zeros(1),
            t=grid.t,
            frames=np.stack(data.frames),
        )
