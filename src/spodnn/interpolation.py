"""sPOD-I: predict at a new parameter by interpolating the training data in mu.

At every time instant, the sPOD amplitudes and shifts of each frame are
interpolated linearly in mu between the training samples; the field is then
rebuilt with spodnn.decomposition.reconstruct. No network is involved.

Ported from paper/Helper.py (my_delta_interpolate, my_interpolated_state).
"""

import numpy as np
from scipy.interpolate import make_interp_spline

from spodnn.decomposition import reconstruct


def interpolate_in_mu(values, mu_train, mu_test, nt):
    """Linear interpolation in mu at every time instant.

    Args:
        values: (rows, n_train * nt), samples stacked along the columns as in
            SyntheticData (columns k*nt .. (k+1)*nt - 1 belong to mu_train[k]).
        mu_train: Training parameters, increasing.
        mu_test: Parameters to predict at; must lie within [min, max] of mu_train.
        nt: Number of time instants per sample.

    Returns:
        (rows, n_test * nt), stacked the same way.
    """
    mu_train, mu_test = np.asarray(mu_train, dtype=float), np.asarray(mu_test, dtype=float)
    if mu_test.min() < mu_train.min() or mu_test.max() > mu_train.max():
        raise ValueError(f"mu_test {mu_test} outside the training range "
                         f"[{mu_train.min()}, {mu_train.max()}]; sPOD-I does not extrapolate")
    rows = values.shape[0]
    per_sample = values.reshape(rows, len(mu_train), nt)
    interpolated = make_interp_spline(mu_train, per_sample, k=1, axis=1)(mu_test)
    return interpolated.reshape(rows, len(mu_test) * nt)


def spod_i(spod_result, shifts_train, mu_train, mu_test, grid):
    """sPOD-I prediction at mu_test.

    Args:
        spod_result: sPOD of the training data (spodnn.decomposition.spod).
        shifts_train: Training shifts per frame, each (1, n_train * nt).
        mu_train, mu_test: As in interpolate_in_mu.
        grid: Grid of the case (needs Nt, Nx, L, dx).

    Returns:
        (q, amplitudes, shifts): the field (Nx, n_test * nt), and the interpolated
        amplitudes and shifts per frame.
    """
    amplitudes = [interpolate_in_mu(a, mu_train, mu_test, grid.Nt) for a in spod_result.time_amplitudes]
    shifts = [interpolate_in_mu(s, mu_train, mu_test, grid.Nt) for s in shifts_train]
    q = reconstruct(spod_result.modes, amplitudes, shifts, grid)
    return q, amplitudes, shifts
