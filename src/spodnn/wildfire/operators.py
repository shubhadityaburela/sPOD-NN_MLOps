"""Finite-difference operators for the wildfire model on a periodic grid.

Fields on an Nx x Ny grid are flattened in Fortran order (x fastest), so the
2D operators are Kronecker products of the 1D ones.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

# 5th-order upwind stencils for the first derivative, at offsets -3..3.
GRAD_STENCIL = np.array([-2, 15, -60, 20, 30, -3, 0]) / 60
DIV_STENCIL = np.array([0, 3, -30, -20, 60, -15, 2]) / 60


def d1_periodic(stencil, n, h):
    """Periodic first-derivative matrix (n x n, CSR) for a centred stencil of odd length on spacing h."""
    if n < len(stencil):
        raise ValueError(f"need at least {len(stencil)} grid points for this stencil, got {n}")
    half = (len(stencil) - 1) // 2
    diagonals, offsets = [], []
    for k, coeff in zip(range(-half, half + 1), stencil, strict=True):
        if coeff == 0:
            continue
        diagonals.append(np.full(n - abs(k), coeff))
        offsets.append(k)
        if k != 0:  # periodic wrap-around in the opposite corner
            diagonals.append(np.full(abs(k), coeff))
            offsets.append(k + n if k < 0 else k - n)
    return sparse.diags(diagonals, offsets, shape=(n, n), format="csr") / h


@dataclass(frozen=True)
class Operators:
    """Operators of the wildfire RHS: the x-gradient (for the wind) and the Laplacian (diffusion)."""

    grad_x: sparse.csc_matrix
    laplace: sparse.csc_matrix


def build_operators(nx, ny, dx, dy):
    """Assemble the x-gradient and the Laplacian on an nx x ny periodic grid (ny == 1 is 1D)."""
    eye_y = sparse.eye(ny, format="csc")
    grad_x = sparse.kron(eye_y, d1_periodic(GRAD_STENCIL, nx, dx), format="csc")
    div_x = sparse.kron(eye_y, d1_periodic(DIV_STENCIL, nx, dx), format="csc")
    laplace = div_x @ grad_x
    if ny > 1:
        eye_x = sparse.eye(nx, format="csc")
        grad_y = sparse.kron(d1_periodic(GRAD_STENCIL, ny, dy), eye_x, format="csc")
        div_y = sparse.kron(d1_periodic(DIV_STENCIL, ny, dy), eye_x, format="csc")
        laplace = laplace + div_y @ grad_y
    return Operators(grad_x=grad_x, laplace=laplace)
