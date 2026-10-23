"""Tests for spodnn.decomposition.

Regression tests run the paper's configuration and compare with the golden
fingerprints. Property tests check facts that hold for any number of modes a
user picks.
"""

from itertools import pairwise

import numpy as np
import pytest
from fingerprints import assert_matches_amplitudes, assert_matches_fingerprint
from numpy.testing import assert_allclose
from spod.sPOD_tools import shifted_rPCA

import spodnn.decomposition as decomposition
from spodnn.decomposition import align_signs, orientation, pod, spod
from spodnn.synthetic import Grid, generate

# Tolerances from evidence: a deterministic LAPACK SVD reproduces the reference to
# ~1e-14; seeded sPOD to ~5e-13 (with margin for other platforms' BLAS).
RTOL_POD = 1e-12
RTOL_SPOD = 1e-10


# pod_result and spod_result (the paper's configuration) are fixtures in conftest.py.


@pytest.fixture(scope="module")
def reference(golden):
    return golden["intermediates"]["decomposition"]


# --- POD: regression -------------------------------------------------------


def test_pod_reconstruction(pod_result, reference):
    assert_matches_fingerprint(
        pod_result.reconstruction, reference["pod_reconstruction"], RTOL_POD, "pod_reconstruction"
    )


# np.linalg.svd fixes no signs, so POD mode signs depend on the numpy/LAPACK build.
# Aligning to the paper's orientation is how the paper's POD-NN will be run, so the
# signs are compared after alignment; the magnitudes don't depend on it.


@pytest.fixture(scope="module")
def pod_aligned(pod_result, reference):
    return align_signs(pod_result, reference["pod_amplitudes_train"]["orientation"])


def test_pod_amplitudes_train(pod_aligned, reference):
    assert_matches_amplitudes(
        pod_aligned.time_amplitudes, reference["pod_amplitudes_train"], RTOL_POD, "pod_amplitudes_train"
    )


def test_pod_amplitudes_test(pod_aligned, reference):
    # The flips are decided on the training amplitudes only; the test amplitudes
    # matching too shows the flips are consistent.
    assert_matches_amplitudes(
        pod_aligned.time_amplitudes_test, reference["pod_amplitudes_test"], RTOL_POD, "pod_amplitudes_test"
    )


def test_align_signs_keeps_reconstruction(pod_result, pod_aligned):
    assert_allclose(
        pod_aligned.modes @ pod_aligned.time_amplitudes,
        pod_result.reconstruction,
        rtol=0,
        atol=1e-12 * np.abs(pod_result.reconstruction).max(),
    )


def test_align_signs_to_own_orientation_changes_nothing(pod_result):
    same = align_signs(pod_result, orientation(pod_result.time_amplitudes))
    assert np.array_equal(same.modes, pod_result.modes)
    assert np.array_equal(same.time_amplitudes_test, pod_result.time_amplitudes_test)


# --- POD: properties for any n_modes ----------------------------------------


@pytest.mark.parametrize("n_modes", [1, 5, 18, 50])
def test_pod_modes_are_orthonormal(train_set, test_set, n_modes):
    result = pod(train_set.q, test_set.q, n_modes=n_modes)
    n_space, n_snapshots = train_set.q.shape
    assert result.modes.shape == (n_space, n_modes)
    assert result.time_amplitudes.shape == (n_modes, n_snapshots)
    assert result.time_amplitudes_test.shape == (n_modes, test_set.q.shape[1])
    assert_allclose(result.modes.T @ result.modes, np.eye(n_modes), atol=1e-10)


def test_pod_error_decreases_with_modes(train_set, test_set):
    """Eckart-Young: adding modes never increases the truncation error."""
    q = train_set.q
    errors = [np.linalg.norm(q - pod(q, test_set.q, n_modes=n).reconstruction) for n in (1, 2, 5, 10, 18, 50)]
    assert all(later <= earlier * (1 + 1e-12) for earlier, later in pairwise(errors)), errors


# --- sPOD: regression (runs 300 iterations, ~2 minutes) --------------------


@pytest.mark.slow
@pytest.mark.parametrize("frame", [1, 2])
def test_spod_frame_reconstruction(spod_result, reference, frame):
    k = frame - 1
    assert_matches_fingerprint(
        spod_result.modes[k] @ spod_result.time_amplitudes[k],
        reference[f"frame{frame}_reconstruction"],
        RTOL_SPOD,
        f"frame{frame}_reconstruction",
    )


@pytest.mark.slow
@pytest.mark.parametrize("frame", [1, 2])
def test_spod_frame_amplitudes_train(spod_result, reference, frame):
    assert_matches_amplitudes(
        spod_result.time_amplitudes[frame - 1],
        reference[f"frame{frame}_amplitudes_train"],
        RTOL_SPOD,
        f"frame{frame}_amplitudes_train",
    )


@pytest.mark.slow
@pytest.mark.parametrize("frame", [1, 2])
def test_spod_frame_amplitudes_test(spod_result, reference, frame):
    assert_matches_amplitudes(
        spod_result.time_amplitudes_test[frame - 1],
        reference[f"frame{frame}_amplitudes_test"],
        RTOL_SPOD,
        f"frame{frame}_amplitudes_test",
    )


@pytest.mark.slow
def test_spod_offline_error(spod_result, golden):
    assert_allclose(spod_result.offline_error, golden["values"]["Error for truncated sPOD recons."], rtol=RTOL_SPOD)


def test_interpolation_error(train_set, test_set, golden):
    # spod() computes this before iterating, so one iteration is enough here.
    result = spod(train_set.q, test_set.frames, train_set.transforms, n_modes=1, n_iter=1)
    # The paper printed it as %4.4e (5 significant digits), so that is all we can compare.
    assert_allclose(result.interpolation_error, golden["values"]["Transformation interpolation error"], rtol=2e-5)


# --- sPOD: properties --------------------------------------------------------


def test_spod_shapes_for_other_n_modes(train_set, test_set):
    n_modes = 3
    result = spod(train_set.q, test_set.frames, train_set.transforms, n_modes=n_modes, n_iter=10)
    n_space, n_snapshots = train_set.q.shape
    assert len(result.modes) == len(result.time_amplitudes) == len(result.time_amplitudes_test) == 2
    for U, A, A_test in zip(result.modes, result.time_amplitudes, result.time_amplitudes_test, strict=True):
        assert U.shape == (n_space, n_modes)
        assert A.shape == (n_modes, n_snapshots)
        assert A_test.shape == (n_modes, test_set.q.shape[1])
    assert result.reconstruction.shape == train_set.q.shape
    assert np.isfinite(result.offline_error)


# --- sPOD: the options the wildfire cases use --------------------------------


@pytest.fixture(scope="module")
def small():
    """A small synthetic case: three training mu and one test mu on a coarse grid."""
    small_grid = Grid(Nx=64, Nt=32)
    return generate(np.array([0.1, 0.2, 0.3]), small_grid), generate(np.array([0.25]), small_grid)


@pytest.fixture
def rpca_calls(monkeypatch):
    """Record the keyword arguments spod() passes to shifted_rPCA (the run itself is real)."""
    calls = []

    def recording(q, trafos, **kwargs):
        calls.append(kwargs)
        return shifted_rPCA(q, trafos, **kwargs)

    monkeypatch.setattr(decomposition, "shifted_rPCA", recording)
    return calls


def test_matrix_penalty_and_default_nmodes_max(small, rpca_calls):
    train, test = small
    spod(train.q, test.frames, train.transforms, n_modes=3, n_iter=2, mu_factor=0.5, lamda_factor=4.0)
    n, m = train.q.shape
    assert_allclose(rpca_calls[0]["mu"], n * m / (4 * np.abs(train.q).sum()) * 0.5, rtol=1e-14)
    assert_allclose(rpca_calls[0]["lambd"], 4.0 / np.sqrt(max(n, m)), rtol=1e-14)
    assert rpca_calls[0]["nmodes_max"] == 3 + 20  # the paper's synthetic setting


def test_grid_penalty_uses_the_grid_size(small, rpca_calls):
    train, test = small
    spod(train.q, test.frames, train.transforms, n_modes=3, n_iter=2, mu_factor=0.5, lamda_factor=4.0,
         penalty="grid", grid_shape=(16, 40), nmodes_max=7, eps=1e-4)  # fmt: skip
    n = train.q.shape[0]
    assert_allclose(rpca_calls[0]["mu"], n / (4 * np.abs(train.q).sum()) * 0.5, rtol=1e-14)
    assert_allclose(rpca_calls[0]["lambd"], 4.0 / np.sqrt(40), rtol=1e-14)
    assert (rpca_calls[0]["nmodes_max"], rpca_calls[0]["eps"]) == (7, 1e-4)


def test_n_modes_none_keeps_the_rank_spod_finds(small):
    train, test = small
    # mu_factor 0.05: on this coarse grid the paper's 0.005 needs ~30 iterations to get going.
    kept = spod(train.q, test.frames, train.transforms, n_modes=None, n_iter=5, mu_factor=0.05, nmodes_max=6)
    truncated = spod(train.q, test.frames, train.transforms, n_modes=1, n_iter=5, mu_factor=0.05, nmodes_max=6)
    assert truncated.ranks == [1, 1]
    for U, A, A_test, rank in zip(kept.modes, kept.time_amplitudes, kept.time_amplitudes_test, kept.ranks, strict=True):
        assert 1 <= rank <= 6
        assert U.shape[1] == A.shape[0] == A_test.shape[0] == rank
    assert sum(kept.ranks) > sum(truncated.ranks)


def test_without_test_frames_there_are_no_test_amplitudes(small):
    train, _ = small
    result = spod(train.q, None, train.transforms, n_modes=2, n_iter=4, mu_factor=0.05)
    assert result.time_amplitudes_test is None
    assert len(result.error_history) == 4  # one residual per iteration
    assert result.error_history[-1] < result.error_history[0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"penalty": "frobenius"}, "penalty must be"),
        ({"penalty": "grid"}, "needs grid_shape"),
        ({"n_modes": None}, "needs an explicit nmodes_max"),
    ],
    ids=["unknown-penalty", "grid-without-shape", "keep-rank-without-nmodes_max"],
)
def test_bad_arguments_are_rejected(small, kwargs, match):
    train, test = small
    with pytest.raises(ValueError, match=match):
        spod(train.q, test.frames, train.transforms, **{"n_modes": 2, "n_iter": 1, **kwargs})
