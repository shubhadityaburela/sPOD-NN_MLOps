"""Compare arrays with the fingerprints stored in the golden file."""

import numpy as np
from numpy.testing import assert_allclose


def assert_matches_fingerprint(array, fp, rtol, name=""):
    a = np.asarray(array, dtype=float)
    assert list(a.shape) == fp["shape"], f"{name}: shape {a.shape} != {fp['shape']}"
    assert_allclose(np.linalg.norm(a), fp["norm"], rtol=rtol, err_msg=f"{name}: norm")
    assert_allclose(np.sum(a), fp["sum"], rtol=0, atol=1e-9 * fp["norm"], err_msg=f"{name}: sum")
    assert_allclose(np.max(np.abs(a)), fp["abs_max"], rtol=0, atol=1e-9 * fp["norm"], err_msg=f"{name}: abs_max")
    assert_allclose(np.mean(np.abs(a)), fp["abs_mean"], rtol=0, atol=1e-9 * fp["norm"], err_msg=f"{name}: abs_mean")

    