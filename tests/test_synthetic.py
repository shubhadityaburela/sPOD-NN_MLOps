"""Unit tests for spodnn.synthetic against the reference data fingerprints."""

import pytest

from spodnn.synthetic import generate, Grid
from fingerprints import assert_matches_fingerprint


TRAIN_MU = [0.1, 0.15, 0.2, 0.25, 0.3]
TEST_MU = [0.23]
RTOL = 1e-12

grid = Grid()  # create a grid instance for the tests


@pytest.fixture(scope="module")
def train_data():
    return generate(TRAIN_MU, grid, n_terms=8)        # built once for all tests in this file


@pytest.fixture(scope="module")
def test_data():
    return generate(TEST_MU, grid, n_terms=8)     # built once for all tests in this file


def test_q_train_matches_reference(train_data, golden):
    assert_matches_fingerprint(train_data.q, golden["intermediates"]["data"]["q_train"], RTOL, "q_train")

def test_q_test_matches_reference(test_data, golden):
    assert_matches_fingerprint(test_data.q, golden["intermediates"]["data"]["q_test"], RTOL, "q_test")

def test_q1_test_matches_reference(test_data, golden):
    assert_matches_fingerprint(test_data.frames[0], golden["intermediates"]["data"]["q1_test"], RTOL, "q1_test")

def test_q2_test_matches_reference(test_data, golden):
    assert_matches_fingerprint(test_data.frames[1], golden["intermediates"]["data"]["q2_test"], RTOL, "q2_test")

def test_shifts_train_frame1_matches_reference(train_data, golden):
    assert_matches_fingerprint(train_data.shifts[0], golden["intermediates"]["data"]["shifts_train_frame1"], RTOL, "shifts_train_frame1")

def test_shifts_train_frame2_matches_reference(train_data, golden):
    assert_matches_fingerprint(train_data.shifts[1], golden["intermediates"]["data"]["shifts_train_frame2"], RTOL, "shifts_train_frame2")

def test_shifts_test_frame1_matches_reference(test_data, golden):
    assert_matches_fingerprint(test_data.shifts[0], golden["intermediates"]["data"]["shifts_test_frame1"], RTOL, "shifts_test_frame1")

def test_shifts_test_frame2_matches_reference(test_data, golden):
    assert_matches_fingerprint(test_data.shifts[1], golden["intermediates"]["data"]["shifts_test_frame2"], RTOL, "shifts_test_frame2")

def test_params_train_matches_reference(train_data, golden):
    assert_matches_fingerprint(train_data.params, golden["intermediates"]["data"]["params_train"], RTOL, "params_train")

def test_params_test_matches_reference(test_data, golden):
    assert_matches_fingerprint(test_data.params, golden["intermediates"]["data"]["params_test"], RTOL, "params_test")
