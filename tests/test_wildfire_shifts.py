"""Tests for spodnn.wildfire.shifts."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from spodnn.wildfire.shifts import compress_shifts, expand_shifts, shifts_1d, shifts_2d, shifts_2d_edges

N_R, N_THETA, NT = 7, 11, 5


@pytest.fixture
def radial():
    return np.random.default_rng(0).uniform(0, 50, size=(N_THETA, NT))


@pytest.fixture
def delta(radial):
    # Built the way Shifts_2D builds it: a list of two frames, radial shift repeated over r.
    delta = [np.zeros((2, N_R, N_THETA, NT)), np.zeros((2, N_R, N_THETA, NT))]
    for n in range(NT):
        delta[0][0, ..., n] = np.repeat(radial[None, :, n], N_R, axis=0)
    return delta


def test_round_trip_is_exact(delta):
    assert_array_equal(expand_shifts(compress_shifts(delta), N_R), np.asarray(delta))


def test_compress_keeps_only_the_radial_shift(delta, radial):
    assert_array_equal(compress_shifts(delta), radial)


def test_expanded_frames_have_the_spod_layout(delta, radial):
    expanded = expand_shifts(radial, N_R)
    for f in range(2):
        assert_array_equal(expanded[f].reshape(2, -1, NT), delta[f].reshape(2, -1, NT))


def test_expand_keeps_dtype(radial):
    assert expand_shifts(radial.astype(np.float32), N_R).dtype == np.float32


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: d[1].__setitem__((0, 0, 0, 0), 1.0),  # frame 2 not zero
        lambda d: d[0].__setitem__((1, 0, 0, 0), 1.0),  # angular shift not zero
        lambda d: d[0].__setitem__((0, 3, 0, 0), -1.0),  # radial shift varies with r
    ],
)
def test_compress_refuses_to_lose_information(delta, corrupt):
    corrupt(delta)
    with pytest.raises(ValueError):
        compress_shifts(delta)


def test_compress_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected shape"):
        compress_shifts(np.zeros((2, 3, 4, 5)))


# Shifts measured on the small solver runs from conftest.py.


def test_1d_shifts_mirror_each_other(fire_1d):
    # No wind: the left front moves left exactly as fast as the right one moves right.
    cfg, sol = fire_1d
    delta = shifts_1d(sol.q, sol.x)
    assert delta.shape == (3, sol.q.shape[1])
    assert not np.any(delta[1])  # the middle frame stays put
    assert np.all(delta[0] <= 0) and np.all(delta[2] >= 0)
    assert_array_equal(delta[0], -delta[2])


def test_1d_shifts_shrink_towards_the_reference(fire_1d):
    # Shifts are measured relative to the last snapshot, so they decrease over time.
    _, sol = fire_1d
    right = shifts_1d(sol.q, sol.x)[2]
    assert np.all(np.diff(right) <= 1e-9)
    assert right[0] > 10 * abs(right[-1])


def test_2d_shift_is_radial_and_shrinks(fire_2d):
    _, sol = fire_2d
    delta = shifts_2d(sol.q, sol.x, sol.y)
    assert delta.shape == (2, 2, sol.q.shape[1])
    assert not np.any(delta[1]) and not np.any(delta[0, 1])  # only frame 1, radial
    assert delta[0, 0, 0] > 10 * abs(delta[0, 0, -1])


@pytest.fixture(scope="module")
def edge_shifts(fire_2d, fire_2d_ww):
    return {name: shifts_2d_edges(sol.q, sol.x, sol.y) for name, (_, sol) in [("2d", fire_2d), ("2d_ww", fire_2d_ww)]}


def test_edge_shifts_are_compact(fire_2d_ww, edge_shifts):
    cfg, sol = fire_2d_ww
    radial = edge_shifts["2d_ww"]
    assert radial.shape == (cfg.grid.Ny, sol.q.shape[1])  # (N_theta, Nt)
    assert expand_shifts(radial, cfg.grid.Nx).shape == (2, 2, cfg.grid.Nx, cfg.grid.Ny, sol.q.shape[1])


@pytest.mark.parametrize("name", ["2d", "2d_ww"])
def test_edge_shifts_are_zero_at_the_reference_and_positive_before(edge_shifts, name):
    radial = edge_shifts[name]
    assert np.all(radial[:, -1] == 0)  # the last snapshot is the reference front
    assert radial.min() > -1e-12
    assert radial[:, 0].mean() > 10


def test_edge_shifts_see_the_shape_of_the_front(edge_shifts):
    # Without wind the front is a circle: the same shift at every angle. Wind makes it lopsided.
    spread = {name: radial[:, 0].std() / radial[:, 0].mean() for name, radial in edge_shifts.items()}
    assert spread["2d"] < 0.05  # measured: 0.03
    assert spread["2d_ww"] > 0.3  # measured: 0.52
