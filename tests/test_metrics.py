"""Tests for spodnn.metrics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spodnn.metrics import error_inf_per_time, error_per_time, relative_error


@pytest.fixture
def q():
    return np.random.default_rng(0).normal(size=(50, 20))


@pytest.mark.parametrize("ord", [None, 1, np.inf])
def test_perfect_prediction_has_zero_error(q, ord):
    assert relative_error(q, q, ord) == 0
    assert np.all(error_per_time(q, q) == 0)
    assert np.all(error_inf_per_time(q, q) == 0)


@pytest.mark.parametrize("ord", [None, 1, np.inf])
def test_predicting_zero_gives_error_one(q, ord):
    assert_allclose(relative_error(q, np.zeros_like(q), ord), 1.0)


def test_errors_are_relative(q):
    # Scaling field and prediction together doesn't change any error.
    q_pred = q + 0.1
    assert_allclose(relative_error(1e3 * q, 1e3 * q_pred), relative_error(q, q_pred))
    assert_allclose(error_per_time(1e3 * q, 1e3 * q_pred), error_per_time(q, q_pred))
    assert_allclose(error_inf_per_time(1e3 * q, 1e3 * q_pred), error_inf_per_time(q, q_pred))


def test_per_time_errors_have_one_value_per_column(q):
    assert error_per_time(q, q + 1).shape == (20,)
    assert error_inf_per_time(q, q + 1).shape == (20,)


def test_mean_square_of_per_time_error_is_the_frobenius_error(q):
    q_pred = q + np.random.default_rng(1).normal(scale=0.1, size=q.shape)
    assert_allclose(np.sqrt(np.mean(error_per_time(q, q_pred) ** 2)), relative_error(q, q_pred))


def test_inf_error_is_worst_point_over_peak():
    q = np.array([[1.0, 2.0], [-4.0, 1.0]])
    q_pred = np.array([[1.5, 2.0], [-4.0, 0.0]])
    assert_allclose(error_inf_per_time(q, q_pred), [0.5 / 4.0, 1.0 / 2.0])
