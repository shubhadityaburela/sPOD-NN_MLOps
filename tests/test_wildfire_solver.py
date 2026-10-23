"""Tests for spodnn.wildfire.solver, on the small grids from conftest.py."""

import math

import numpy as np
import pytest
from conftest import TEST_BETA, small_wildfire
from numpy.testing import assert_allclose, assert_array_equal

from spodnn.wildfire.solver import solve


def fields(cfg, sol):
    """T and S as (Nt, Nx, Ny), using the documented snapshot layout."""
    nx, ny, nt = cfg.grid.Nx, cfg.grid.Ny, sol.q.shape[1]
    if ny == 1:
        return sol.q[:nx].T[:, :, None], sol.q[nx:].T[:, :, None]
    q = sol.q.T.reshape(nt, 2, nx, ny, order="F")
    return q[:, 0], q[:, 1]


@pytest.fixture(params=["fire_1d", "fire_2d", "fire_2d_ww"])
def fire(request):
    return request.getfixturevalue(request.param)


def test_shapes_and_grid(fire):
    cfg, sol = fire
    g = cfg.grid
    assert sol.q.shape == (2 * g.Nx * g.Ny, g.Nt // cfg.step)
    assert sol.x.shape == (g.Nx,)
    assert sol.y.shape == (g.Ny,)
    assert_allclose(sol.x[[0, -1]], [g.Lx / g.Nx, g.Lx])


def test_time_labels_are_the_end_of_each_saved_step(fire):
    # Column k is the state after time step k*step, at time (k*step + 1)*dt.
    cfg, sol = fire
    dx = cfg.grid.Lx / cfg.grid.Nx
    dt = math.sqrt(dx**2 * (1 if cfg.grid.Ny == 1 else 2)) * cfg.physics.cfl / math.sqrt(cfg.physics.c)
    assert sol.t.shape == (cfg.grid.Nt // cfg.step,)
    assert_allclose(sol.t, dt * (np.arange(0, cfg.grid.Nt, cfg.step) + 1), rtol=1e-12)


def test_layout_separates_temperature_and_fuel(fire):
    # A wrong reshape would mix T (up to ~1200) into S (between 0 and 1).
    T, S = fields(*fire)
    assert T.max() > 100
    assert S.min() > 0 and S.max() <= 1


def test_fuel_never_grows(fire):
    _, S = fields(*fire)
    assert np.all(np.diff(S, axis=0) <= 0)


def test_fire_burns_fuel(fire):
    _, S = fields(*fire)
    assert S[-1].min() < 0.01  # burnt out in the middle
    assert S[-1].max() == 1  # untouched far away


def test_same_input_gives_bit_identical_output(fire_1d):
    cfg, sol = fire_1d
    assert_array_equal(solve(cfg.grid, cfg.physics, TEST_BETA, cfg.step).q, sol.q)


def test_1d_fire_spreads_symmetrically(fire_1d):
    # No wind: mirror image about the centre x = Lx/2 (index Nx/2 - 1 on this grid).
    cfg, sol = fire_1d
    T, S = fields(cfg, sol)
    n = cfg.grid.Nx
    i = np.arange(n)
    mirror = (n // 2 - 1 - i) % n
    centred = (n // 2 - 1 + i) % n
    assert_allclose(T[:, mirror], T[:, centred], atol=1e-9)
    assert_allclose(S[:, mirror], S[:, centred], atol=1e-12)


def test_2d_fire_without_wind_is_symmetric_in_x_and_y(fire_2d):
    T, S = fields(*fire_2d)
    assert_allclose(T, T.transpose(0, 2, 1), atol=1e-9)
    assert_allclose(S, S.transpose(0, 2, 1), atol=1e-12)


def _burnt_centre_x(cfg, sol):
    _, S = fields(cfg, sol)
    burnt = (1 - S[-1]).sum(axis=1)
    return (burnt * sol.x).sum() / burnt.sum()


def test_wind_carries_the_fire_downwind(fire_2d, fire_2d_ww):
    # Without wind the burnt area stays in the centre of the 100 x 100 domain (up to the periodic
    # wrap-around); v_x = 0.2 pushes it in +x.
    assert _burnt_centre_x(*fire_2d) == pytest.approx(50, abs=0.5)
    assert _burnt_centre_x(*fire_2d_ww) > 53


def test_higher_activation_energy_leaves_more_fuel():
    cfg = small_wildfire("wildfire1d", Lx=200, Nx=400, Nt=400)
    fuel = [solve(cfg.grid, cfg.physics, beta, cfg.step).q[400:, -1].sum() for beta in (540, 580)]
    assert fuel[0] < fuel[1]
