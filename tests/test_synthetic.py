"""Unit tests for spodnn.synthetic against the reference data fingerprints."""

from fingerprints import assert_matches_fingerprint

RTOL = 1e-12


def test_q_train_matches_reference(train_set, golden):
    assert_matches_fingerprint(train_set.q, golden["intermediates"]["data"]["q_train"], RTOL, "q_train")


def test_q_test_matches_reference(test_set, golden):
    assert_matches_fingerprint(test_set.q, golden["intermediates"]["data"]["q_test"], RTOL, "q_test")


def test_q1_test_matches_reference(test_set, golden):
    assert_matches_fingerprint(test_set.frames[0], golden["intermediates"]["data"]["q1_test"], RTOL, "q1_test")


def test_q2_test_matches_reference(test_set, golden):
    assert_matches_fingerprint(test_set.frames[1], golden["intermediates"]["data"]["q2_test"], RTOL, "q2_test")


def test_shifts_train_frame1_matches_reference(train_set, golden):
    assert_matches_fingerprint(
        train_set.shifts[0], golden["intermediates"]["data"]["shifts_train_frame1"], RTOL, "shifts_train_frame1"
    )


def test_shifts_train_frame2_matches_reference(train_set, golden):
    assert_matches_fingerprint(
        train_set.shifts[1], golden["intermediates"]["data"]["shifts_train_frame2"], RTOL, "shifts_train_frame2"
    )


def test_shifts_test_frame1_matches_reference(test_set, golden):
    assert_matches_fingerprint(
        test_set.shifts[0], golden["intermediates"]["data"]["shifts_test_frame1"], RTOL, "shifts_test_frame1"
    )


def test_shifts_test_frame2_matches_reference(test_set, golden):
    assert_matches_fingerprint(
        test_set.shifts[1], golden["intermediates"]["data"]["shifts_test_frame2"], RTOL, "shifts_test_frame2"
    )


def test_params_train_matches_reference(train_set, golden):
    assert_matches_fingerprint(train_set.params, golden["intermediates"]["data"]["params_train"], RTOL, "params_train")


def test_params_test_matches_reference(test_set, golden):
    assert_matches_fingerprint(test_set.params, golden["intermediates"]["data"]["params_test"], RTOL, "params_test")
