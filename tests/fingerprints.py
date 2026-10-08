"""Compare arrays with the fingerprints stored in the golden file."""

import numpy as np
from numpy.testing import assert_allclose


def assert_matches_fingerprint(array, fp, rtol, name=""):
    """Assert that `array` matches a fingerprint from scripts/make_golden.py.

    The shape must match exactly; norm, abs_max and abs_mean must agree to the
    relative tolerance `rtol`. The sum can be close to zero (positive and negative
    parts cancel), so it also gets an absolute tolerance scaled to the norm.
    `name` labels the array in failure messages.
    """
    a = np.asarray(array, dtype=float)
    assert list(a.shape) == fp["shape"], f"{name}: shape {a.shape} != {fp['shape']}"
    assert_allclose(np.linalg.norm(a), fp["norm"], rtol=rtol, err_msg=f"{name}: norm")
    assert_allclose(np.sum(a), fp["sum"], rtol=rtol, atol=1e-9 * fp["norm"], err_msg=f"{name}: sum")
    assert_allclose(np.max(np.abs(a)), fp["abs_max"], rtol=rtol, atol=1e-9 * fp["norm"], err_msg=f"{name}: abs_max")
    assert_allclose(np.mean(np.abs(a)), fp["abs_mean"], rtol=rtol, atol=1e-9 * fp["norm"], err_msg=f"{name}: abs_mean")

    