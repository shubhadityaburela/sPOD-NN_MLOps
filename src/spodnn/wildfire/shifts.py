"""Shifts of the co-moving frames, measured from the flame front in the snapshots.

Ported from WildlandFire_model/Shifts.py with the arithmetic order kept, so
the shifts match the original:

- shifts_1d: two fronts moving left and right (3 frames, the middle one fixed).
- shifts_2d: one radially symmetric front, tracked along the mid-line.
- shifts_2d_edges: a front of any shape (2D_ww), found by edge detection in
  polar coordinates. Only the radial shift of frame 1 is non-zero and it is the
  same for every radius, so it is returned compactly as (N_theta, Nt):
  400 MB less per parameter value. expand_shifts rebuilds the full array.
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import uniform_filter1d

from spodnn.wildfire.polar import detect_edges, to_polar


def _smooth(shift):
    """Cubic fit through the shift at 0, Nt/4, Nt/2, 3Nt/4 and the end, as in the original."""
    nt = shift.size
    knots = np.asarray([0, nt // 4, nt // 2, 3 * nt // 4, nt])
    values = np.asarray([shift[0], shift[nt // 4], shift[nt // 2], shift[3 * nt // 4], shift[-1]])
    return interpolate.interp1d(knots, values, kind="cubic")(np.arange(0, nt))


def shifts_1d(q, x):
    """Shifts (3, Nt) of the left front, a fixed middle frame and the right front, from 1D snapshots [T; S].

    Each front is where the fuel S changes fastest; shifts are relative to the
    last snapshot, negative to the left.
    """
    nx, nt = x.size, q.shape[1]
    dx = x[1] - x[0]
    left, right = np.zeros(nt), np.zeros(nt)
    for n in range(nt):
        grad = np.diff(q[nx : 2 * nx, n]) / dx
        left[n] = x[np.argmin(grad)]
        right[n] = x[np.argmax(grad)]
    delta = np.zeros((3, nt))
    delta[0] = _smooth(-np.abs(left - left[-1]))
    delta[2] = _smooth(np.abs(right - right[-1]))
    return delta


def _fuel(q, nx, ny):
    """The fuel S as (Nx, Ny, Nt) from 2D snapshots (T and S interleaved per point)."""
    return q.T.reshape(q.shape[1], 2, nx, ny, order="F")[:, 1].transpose(1, 2, 0)


def shifts_2d(q, x, y):
    """Shifts (2 frames, 2 directions, Nt) of a circular front, tracked along the mid-line y = Ny/2.

    Only the radial shift of frame 1 is non-zero.
    """
    nx, ny, nt = x.size, y.size, q.shape[1]
    dx = x[1] - x[0]
    x_c, y_c = x[-1] // 2, y[-1] // 2
    S = _fuel(q, nx, ny)
    front = np.zeros(nt)
    for n in range(nt):
        grad = np.diff(np.squeeze(S[:, ny // 2, n])) / dx
        front[n] = np.sqrt((x[np.argmin(grad)] - x_c) ** 2 + (y_c - y_c) ** 2)
    delta = np.zeros((2, 2, nt))
    delta[0, 0] = _smooth(np.abs(front - front[-1]))
    return delta


def _fill_gaps(front, theta):
    """Interpolate the front radius over angles where no edge was found (radius 0)."""
    found = np.where(front != 0)[0]
    if np.any(found):  # as in the original: also skipped when the only edge is at index 0
        front = np.interp(x=theta, xp=theta[found], fp=front[found])
    return front


def shifts_2d_edges(q, x, y):
    """Radial shift (N_theta, Nt) of frame 1 for a front of any shape (the compact form).

    The fuel S is resampled to polar coordinates, the front is found with Canny
    edge detection, and its radius per angle is compared with the last snapshot,
    then smoothed over 10 angles. expand_shifts(radial, Nx) gives the full array.
    """
    nx, ny, nt = x.size, y.size, q.shape[1]
    S_polar, r, theta, _ = to_polar(_fuel(q, nx, ny), x, y, fill_value=1)
    _, r_grid = np.meshgrid(theta, r)
    edges = detect_edges(S_polar)

    # Snapshots without any edge copy the edges of snapshot `missing`, as in the original.
    missing = sum(not np.any(edges[..., n]) for n in range(nt))
    for n in range(missing):
        edges[..., n] = edges[..., missing]

    reference = _fill_gaps(np.amax(edges[..., -1] * r_grid, axis=0), theta)
    radial = np.zeros((ny, nt))
    for n in range(nt):
        front = _fill_gaps(np.amax(edges[..., n] * r_grid, axis=0), theta)
        radial[:, n] = uniform_filter1d(np.abs(reference - front), size=10, mode="nearest")
    return radial


def compress_shifts(delta):
    """Reduce the full 2D_ww shift array to its radial shift of shape (N_theta, Nt).

    ``delta`` is what ``Shifts_2D`` returns: an array or a list of two frame
    arrays of shape (2, N_r, N_theta, Nt). Raises ``ValueError`` if ``delta``
    holds anything the compact form would lose.
    """
    delta = np.asarray(delta)
    if delta.ndim != 5 or delta.shape[:2] != (2, 2):
        raise ValueError(f"expected shape (2, 2, N_r, N_theta, Nt), got {delta.shape}")
    radial = delta[0, 0, 0]
    if np.any(delta[1]) or np.any(delta[0, 1]):
        raise ValueError("frame 2 and the angular shifts must be zero")
    if not np.array_equal(delta[0, 0], np.broadcast_to(radial, delta.shape[2:])):
        raise ValueError("the radial shift must be the same for every radius")
    return radial.copy()


def expand_shifts(radial, n_r):
    """Rebuild the full shift array of shape (2, 2, n_r, N_theta, Nt) from the radial shift.

    The result equals what ``Shifts_2D`` returns. The sPOD transform of frame
    ``f`` takes ``delta[f].reshape(2, -1, Nt)``.
    """
    radial = np.asarray(radial)
    if radial.ndim != 2:
        raise ValueError(f"expected shape (N_theta, Nt), got {radial.shape}")
    delta = np.zeros((2, 2, n_r, *radial.shape), dtype=radial.dtype)
    delta[0, 0] = radial
    return delta
