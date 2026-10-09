"""Tests for spodnn.interpolation (sPOD-I) and spodnn.decomposition.reconstruct."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import paper
from spodnn.decomposition import reconstruct
from spodnn.interpolation import interpolate_in_mu, spod_i
from spodnn.synthetic import Grid

NT = 4
MU = [0.1, 0.2, 0.3]


@pytest.fixture
def values():
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, len(MU) * NT))  # 3 rows, 3 samples of NT columns each


def test_at_a_training_mu_returns_that_sample(values):
    assert_allclose(interpolate_in_mu(values, MU, [0.2], NT), values[:, NT : 2 * NT], rtol=1e-14)


def test_between_two_mus_is_the_average(values):
    expected = 0.5 * (values[:, :NT] + values[:, NT : 2 * NT])
    assert_allclose(interpolate_in_mu(values, MU, [0.15], NT), expected, rtol=1e-14)


def test_several_test_mus_are_stacked_like_the_input(values):
    out = interpolate_in_mu(values, MU, [0.1, 0.3], NT)
    assert out.shape == (3, 2 * NT)
    assert_allclose(out, np.hstack([values[:, :NT], values[:, 2 * NT :]]), rtol=1e-14)


@pytest.mark.parametrize("mu", [0.05, 0.35])
def test_no_extrapolation(values, mu):
    with pytest.raises(ValueError, match="outside the training range"):
        interpolate_in_mu(values, MU, [mu], NT)


def test_shifts_are_exact(train_set, test_set):
    # The synthetic shifts are mu * t, linear in mu: interpolation reproduces them.
    for train_shift, test_shift in zip(train_set.shifts, test_set.shifts, strict=True):
        out = interpolate_in_mu(train_shift, paper.TRAIN_MU, paper.TEST_MU, Grid().Nt)
        assert_allclose(out, test_shift, rtol=0, atol=1e-15)


@pytest.mark.slow
def test_reconstruct_inverts_the_spod(spod_result, train_set):
    q = reconstruct(spod_result.modes, spod_result.time_amplitudes, train_set.shifts, Grid())
    assert_allclose(q, spod_result.reconstruction, rtol=0, atol=1e-12 * np.abs(q).max())


@pytest.mark.slow
def test_spod_i_predicts_the_test_field(spod_result, train_set, test_set):
    q, amplitudes, shifts = spod_i(spod_result, train_set.shifts, paper.TRAIN_MU, paper.TEST_MU, Grid())
    assert q.shape == test_set.q.shape
    assert len(amplitudes) == len(shifts) == paper.N_FRAMES
    # Sanity bound only: sPOD-I should be accurate on this smooth synthetic case.
    assert np.linalg.norm(q - test_set.q) / np.linalg.norm(test_set.q) < 1e-2
