"""Tests for spodnn.wildfire.operators."""

import numpy as np
import pytest

from spodnn.wildfire.operators import DIV_STENCIL, GRAD_STENCIL, build_operators, d1_periodic


def _max_error(stencil, n):
    x = np.arange(n) * 2 * np.pi / n
    return np.abs(d1_periodic(stencil, n, x[1] - x[0]) @ np.sin(x) - np.cos(x)).max()


@pytest.mark.parametrize("stencil", [GRAD_STENCIL, DIV_STENCIL], ids=["grad", "div"])
def test_first_derivative_is_fifth_order(stencil):
    errors = [_max_error(stencil, n) for n in (32, 64)]
    assert np.log2(errors[0] / errors[1]) == pytest.approx(5, abs=0.2)


@pytest.mark.parametrize("stencil", [GRAD_STENCIL, DIV_STENCIL], ids=["grad", "div"])
def test_first_derivative_of_a_constant_is_zero(stencil):
    assert np.abs(d1_periodic(stencil, 20, 0.5) @ np.ones(20)).max() < 1e-12


def test_stencil_needs_enough_points():
    with pytest.raises(ValueError, match="at least 7 grid points"):
        d1_periodic(GRAD_STENCIL, 6, 1.0)


@pytest.fixture(scope="module")
def grid_2d():
    nx, ny = 48, 32
    x = np.arange(nx) * 2 * np.pi / nx
    y = np.arange(ny) * 2 * np.pi / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ops = build_operators(nx, ny, x[1] - x[0], y[1] - y[0])
    flat = lambda f: f.reshape(-1, order="F")  # noqa: E731 - Fortran order, x fastest
    return ops, xx, yy, flat


def test_laplacian_of_a_wave_is_minus_k_squared_times_the_wave(grid_2d):
    ops, xx, yy, flat = grid_2d
    f = np.sin(xx) * np.sin(2 * yy)
    assert np.abs(ops.laplace @ flat(f) - flat(-5 * f)).max() < 1e-3


def test_x_gradient_ignores_y(grid_2d):
    ops, xx, yy, flat = grid_2d
    assert np.abs(ops.grad_x @ flat(np.sin(yy))).max() < 1e-12
    assert np.abs(ops.grad_x @ flat(np.sin(xx)) - flat(np.cos(xx))).max() < 1e-3


def test_laplacian_conserves_the_total(grid_2d):
    # Periodic diffusion only moves heat around: every column of the Laplacian sums to zero.
    ops, *_ = grid_2d
    assert np.abs(ops.laplace.sum(axis=0)).max() < 1e-12


def test_1d_operators_have_no_y_part():
    ops = build_operators(50, 1, 0.1, 0)
    assert ops.laplace.shape == ops.grad_x.shape == (50, 50)
