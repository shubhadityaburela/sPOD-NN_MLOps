"""Tests for the patches in the vendored sPOD transforms (general 2D shifts, as used by 2D_ww)."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from spod.transforms import compute_general_shift_matrix_numba, meshgrid2D, transforms

from spodnn.wildfire.shifts import expand_shifts

pytestmark = pytest.mark.slow  # numba compiles the general 2D shift on first use (~7 s)

NX, NY, NT = 12, 12, 4  # square: sPOD's meshgrid2D, used by the general shifts, assumes Nx == Ny


@pytest.fixture(scope="module")
def radial_trafo():
    # A purely radial shift field, built the way the 2D_ww case builds it (frame 1 of expand_shifts).
    radial = np.random.default_rng(0).uniform(0, 5, size=(NY, NT))
    shifts = expand_shifts(radial, NX)[0].reshape(2, -1, NT)
    trafo = transforms([NX, NY, 1, NT], [NX - 1, NY - 1], shifts=shifts, dx=[1.0, 1.0], use_scipy_transform=False)
    return trafo, shifts


def with_zeros(shifts):
    """The shift matrices as the unpatched sPOD built them: every stencil entry stored, zeros included."""
    ix, iy = meshgrid2D(np.arange(NX), np.arange(NY))
    mats = []
    for it in range(NT):
        col, row, val = compute_general_shift_matrix_numba(
            shifts[..., it],
            np.asarray([NX - 1, NY - 1]),
            np.asarray([1.0, 1.0]),
            np.asarray([NX, NY]),
            ix.flatten(),
            iy.flatten(),
        )
        mats.append(sparse.coo_matrix((val, (row, col)), shape=(NX * NY, NX * NY)).tocsc())
    return mats


def test_general_shift_matrices_store_no_zeros(radial_trafo):
    trafo, _ = radial_trafo
    for mat in trafo.shifts_pos + trafo.shifts_neg:
        assert np.all(mat.data != 0)


def test_dropping_zeros_leaves_the_shift_bitwise_unchanged(radial_trafo):
    trafo, shifts = radial_trafo
    field = np.random.default_rng(1).standard_normal((NX * NY, NT))
    for patched, original in zip(trafo.shifts_pos, with_zeros(shifts), strict=True):
        assert patched.nnz < original.nnz  # the radial stencil really had zeros to drop
    expected = np.stack([original @ field[:, it] for it, original in enumerate(with_zeros(shifts))], axis=-1)
    assert_array_equal(trafo.apply(field).reshape(NX * NY, NT), expected)
