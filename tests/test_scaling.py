"""Tests for spodnn.scaling.

Regression: the paper's saved scaling files load correctly, scaling the test
parameters reproduces the reference network inputs, and fitting on our ported
training data reproduces the constants the paper saved. Properties: round trips,
the [0, 1] range, save/load, error cases.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import paper
from fingerprints import assert_matches_fingerprint
from spodnn.decomposition import align_signs
from spodnn.scaling import Scaling

RUNS = "paper/DNN_result/synthetic"
SPOD_NPY = f"{RUNS}/training_results_sPOD/2025_06_26__13-27-59/variables/scaling.npy"
POD_NPY = f"{RUNS}/training_results_POD/2025_06_26__14-24-05/variables/scaling.npy"


@pytest.fixture(scope="module")
def repo():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def paper_spod(repo):
    return Scaling.from_paper_npy(repo / SPOD_NPY, n_amplitudes=paper.N_FRAMES * paper.SPOD_MODES)


@pytest.fixture(scope="module")
def paper_pod(repo):
    return Scaling.from_paper_npy(repo / POD_NPY, n_amplitudes=paper.POD_MODES)


# --- the paper's files -------------------------------------------------------

def test_paper_spod_scaling_values(paper_spod):
    assert paper_spod.has_shifts
    assert_allclose([paper_spod.amplitude_min, paper_spod.amplitude_max], [-1774.9142899159344, 1775.5707465424332])
    assert_allclose([paper_spod.shift_min, paper_spod.shift_max], [-0.2994, 0.2994])
    assert_allclose(paper_spod.param_min, [0.0, 0.1])  # rows [t, mu]
    assert_allclose(paper_spod.param_max, [0.998, 0.3])


def test_paper_pod_scaling_has_no_shifts_and_identity_params(paper_pod):
    # The paper's POD-NN was trained on parameters already scaled in place by the
    # sPOD-NN run, so its file stores the identity; see Scaling.from_paper_npy.
    assert not paper_pod.has_shifts
    assert_allclose(paper_pod.param_min, [0.0, 0.0])
    assert_allclose(paper_pod.param_max, [1.0, 1.0])


def test_scaled_test_params_match_reference(paper_spod, test_set, golden):
    scaled = paper_spod.scale_params(test_set.params)
    assert_matches_fingerprint(scaled, golden["intermediates"]["network"]["params_test_scaled"],
                               1e-12, "params_test_scaled")


def test_scale_params_does_not_modify_its_input(paper_spod, test_set):
    params = test_set.params.copy()
    paper_spod.scale_params(params)
    assert np.array_equal(params, test_set.params)


# --- fitting reproduces the paper's constants (needs the sPOD run) ----------

@pytest.mark.slow
def test_fit_reproduces_paper_spod_scaling(spod_result, train_set, paper_spod):
    outputs = np.concatenate(spod_result.time_amplitudes + [s.reshape(1, -1) for s in train_set.shifts])
    fitted = Scaling.fit(outputs, train_set.params, n_amplitudes=paper.N_FRAMES * paper.SPOD_MODES)
    for name in ["amplitude_min", "amplitude_max", "shift_min", "shift_max"]:
        assert_allclose(getattr(fitted, name), getattr(paper_spod, name), rtol=1e-10, err_msg=name)
    assert_allclose(fitted.param_min, paper_spod.param_min, rtol=1e-12)
    assert_allclose(fitted.param_max, paper_spod.param_max, rtol=1e-12)


def test_fit_reproduces_paper_pod_amplitude_scaling(pod_result, train_set, paper_pod, golden):
    # Min/max of the amplitudes depend on the mode signs, so align to the paper's basis first.
    reference = golden["intermediates"]["decomposition"]["pod_amplitudes_train"]["orientation"]
    aligned = align_signs(pod_result, reference)
    fitted = Scaling.fit(aligned.time_amplitudes, train_set.params, n_amplitudes=paper.POD_MODES)
    assert not fitted.has_shifts
    assert_allclose([fitted.amplitude_min, fitted.amplitude_max],
                    [paper_pod.amplitude_min, paper_pod.amplitude_max], rtol=1e-12)


# --- properties --------------------------------------------------------------

@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    outputs = rng.normal(size=(5, 40)) * [[3], [3], [3], [0.1], [0.1]]  # 3 amplitudes, 2 shifts
    params = np.vstack([np.linspace(0, 1, 40), np.full(40, 0.2) + rng.uniform(0, 0.1, 40)])
    return outputs, params, Scaling.fit(outputs, params, n_amplitudes=3)


def test_fitted_training_data_spans_unit_interval(toy):
    outputs, params, scaling = toy
    for scaled, rows in [(scaling.scale_outputs(outputs), [slice(0, 3), slice(3, 5)]),
                         (scaling.scale_params(params), [slice(0, 1), slice(1, 2)])]:
        for r in rows:
            assert_allclose([scaled[r].min(), scaled[r].max()], [0.0, 1.0], atol=1e-15)


def test_round_trips(toy):
    outputs, params, scaling = toy
    assert_allclose(scaling.unscale_outputs(scaling.scale_outputs(outputs)), outputs, rtol=1e-13, atol=1e-15)
    assert_allclose(scaling.unscale_params(scaling.scale_params(params)), params, rtol=1e-13, atol=1e-15)


def test_unscale_outputs_keeps_float32(toy):
    outputs, _, scaling = toy
    predicted = scaling.scale_outputs(outputs).astype(np.float32)  # the network predicts in float32
    assert scaling.unscale_outputs(predicted).dtype == np.float32


def test_save_load_round_trip(toy, paper_pod, tmp_path):
    # tmp_path is a built-in pytest fixture: a fresh temporary directory per test.
    for scaling in [toy[2], paper_pod]:
        path = tmp_path / "scaling.json"
        scaling.save(path)
        loaded = Scaling.load(path)
        for name in ["n_amplitudes", "amplitude_min", "amplitude_max", "shift_min", "shift_max"]:
            assert getattr(loaded, name) == getattr(scaling, name), name
        assert np.array_equal(loaded.param_min, scaling.param_min)
        assert np.array_equal(loaded.param_max, scaling.param_max)


def test_fit_rejects_constant_data():
    with pytest.raises(ValueError, match="constant"):
        Scaling.fit(np.ones((2, 10)), np.vstack([np.linspace(0, 1, 10)] * 2), n_amplitudes=2)


def test_wrong_number_of_output_rows_is_rejected(toy, paper_pod):
    _, _, with_shifts = toy
    with pytest.raises(ValueError):
        with_shifts.unscale_outputs(np.zeros((3, 4)))  # 3 amplitude rows but no shift rows
    with pytest.raises(ValueError):
        paper_pod.unscale_outputs(np.zeros((20, 4)))  # POD-NN has exactly 18 outputs
