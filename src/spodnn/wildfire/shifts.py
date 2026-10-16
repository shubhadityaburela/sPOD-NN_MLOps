"""Compact storage for the 2D_ww shifts.

``Shifts_2D(..., edge_detect=True)`` returns an array of shape
(2 frames, 2 directions, N_r, N_theta, Nt), but by problem design only one
slice carries information: the radial shift of frame 1, which is the same for
every radius. Frame 2 and the angular direction are all zeros. Storing that
slice of shape (N_theta, Nt) instead of the full array shrinks the shifts from
400 MB to 200 KB per parameter value.
"""

import numpy as np


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
