"""Tests for spodnn.wildfire.shifts."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from spodnn.wildfire.shifts import compress_shifts, expand_shifts

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
