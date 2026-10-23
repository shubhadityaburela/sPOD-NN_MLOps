"""Wildland fire model: temperature T and fuel supply S on a periodic grid.

    dT/dt = k ΔT - v_x ∂T/∂x - γ_s·γ T + γ H(T) S exp(-β / T)
    dS/dt =                        - γ_s H(T) S exp(-β / T)

solved with RK4. Ported from WildlandFire_model/Wildfire.py: the arithmetic is
kept in the same order, so the snapshots match the original bit for bit.
"""

import math
from dataclasses import dataclass

import numpy as np

from spodnn.config import WildfireGridConfig, WildfirePhysicsConfig
from spodnn.wildfire.operators import Operators, build_operators

EPSILON = 1e-6  # keeps exp(-β / T) finite at T = 0


@dataclass(frozen=True)
class WildfireSolution:
    """Snapshots of one solver run.

    q has one column per saved time step and 2·Nx·Ny rows. 1D: [T; S] stacked.
    2D: T and S interleaved per grid point (T_0, S_0, T_1, S_1, ...), points in
    Fortran order; read it back with q.T.reshape(Nt, 2, Nx, Ny, order="F").
    t[k] is the time of column k: the state after time step k·step, at (k·step + 1)·dt.
    """

    q: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray


def initial_state(x, y, grid):
    """Gaussian temperature bump in the centre of the domain, fuel everywhere: the [T; S] state vector."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if grid.Ny == 1:
        T = 1200 * np.exp(-((xx - grid.Lx / 2) ** 2) / 200)
    else:
        T = 1200 * np.exp(-(((xx - grid.Lx / 2) ** 2) / 200 + ((yy - grid.Ly / 2) ** 2) / 200))
    S = np.ones_like(T)
    return np.concatenate([T.reshape(-1, order="F"), S.reshape(-1, order="F")])


def rhs(q, ops: Operators, wind, physics: WildfirePhysicsConfig, beta):
    """Time derivative of the [T; S] state vector."""
    n = q.size // 2
    T, S = q[:n], q[n:]
    arrhenius_activate = (T > 0).astype(int)  # reaction only where T > 0
    reaction = np.exp(-beta / (T + EPSILON))

    coeff_source = physics.TempRisePerSecond * physics.ScaledHeatTransferCoeff
    Tdot = (
        physics.ThermalDiffusivity * ops.laplace.dot(T)
        - wind.dot(T)
        - coeff_source * T
        + physics.TempRisePerSecond * arrhenius_activate * S * reaction
    )
    Sdot = -physics.PreExponentialFactor * arrhenius_activate * S * reaction
    return np.concatenate([Tdot, Sdot])


def rk4_step(q, dt, *args):
    """One classical Runge-Kutta 4 step; args are passed on to rhs."""
    k1 = rhs(q, *args)
    k2 = rhs(q + dt / 2 * k1, *args)
    k3 = rhs(q + dt / 2 * k2, *args)
    k4 = rhs(q + dt * k3, *args)
    return q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def solve(grid: WildfireGridConfig, physics: WildfirePhysicsConfig, beta, step) -> WildfireSolution:
    """Run the solver for Nt time steps and keep every step-th state."""
    x = np.arange(1, grid.Nx + 1) * grid.Lx / grid.Nx
    dx = x[1] - x[0]
    if grid.Ny == 1:
        y, dy = np.zeros(1), 0
    else:
        y = np.arange(1, grid.Ny + 1) * grid.Ly / grid.Ny
        dy = y[1] - y[0]
    dt = np.sqrt(dx**2 + dy**2) * physics.cfl / math.sqrt(physics.c)

    ops = build_operators(grid.Nx, grid.Ny, dx, dy)
    wind = physics.v_x * ops.grad_x  # constant in time: built once, not in every RHS call

    n = grid.Nx * grid.Ny
    q = initial_state(x, y, grid)
    saved = np.empty((2 * n, grid.Nt // step))
    for i in range(grid.Nt):
        q = rk4_step(q, dt, ops, wind, physics, beta)
        if i % step == 0:
            # 2D stores T and S interleaved per point, like the original (see WildfireSolution).
            saved[:, i // step] = q if grid.Ny == 1 else q.reshape(2, n).T.ravel()

    t = dt * (np.arange(0, grid.Nt, step) + 1)
    return WildfireSolution(q=saved, x=x, y=y, t=t)
