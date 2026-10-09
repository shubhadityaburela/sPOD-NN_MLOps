"""Error measures for predicted fields, as defined in the paper.

All functions take the true field q and a prediction q_pred of shape (Nx, Nt):
one column per time instant. Ported from OnlinePredictionAnalysis in
paper/synthetic_sup.py.
"""

import numpy as np


def relative_error(q, q_pred, ord=None):
    """||q - q_pred|| / ||q|| with numpy's matrix norm `ord`.

    ord=None is the Frobenius norm (the paper's main error). ord=1 and ord=np.inf
    are matrix norms, not entrywise ones: ord=1 is the largest column sum (the worst
    time instant's L1 norm in space), ord=np.inf the largest row sum.
    """
    return np.linalg.norm(q - q_pred, ord) / np.linalg.norm(q, ord)


def error_per_time(q, q_pred):
    """E_j = ||q_j - q_pred_j||_2 / sqrt(mean_j ||q_j||_2^2), one value per time instant j.

    Normalised by the field's mean energy, so sqrt(mean_j E_j^2) equals the
    relative Frobenius error.
    """
    return np.linalg.norm(q - q_pred, axis=0) / np.sqrt(np.mean(np.sum(q**2, axis=0)))


def error_inf_per_time(q, q_pred, eps=1e-16):
    """E_j,inf = max_i |q_ij - q_pred_ij| / max_i |q_ij|: the worst point, relative to the peak, per time instant."""
    return np.max(np.abs(q - q_pred), axis=0) / (np.max(np.abs(q), axis=0) + eps)
