"""Offline decomposition of snapshot data: POD and shifted POD (sPOD).

POD finds one linear basis for all snapshots; sPOD (spod.shifted_rPCA) splits the
data into co-moving frames, each with its own low-rank basis, which suits
transport-dominated fields. Both return the basis ("modes"), the time amplitudes
of the training snapshots and the projection of test snapshots onto the basis.

Ported from paper/synthetic_sup.py (__init__, lines 47-111).

Sign convention: a singular vector is only defined up to sign. sPOD's
randomized_svd fixes signs itself (sklearn's flip_sign), but np.linalg.svd does
not, so POD signs can differ between numpy/LAPACK builds. That is harmless as
long as a trained network is always used with the basis it was trained on; to run
a network trained on another basis (the paper's POD-NN), use align_signs().
"""

from dataclasses import dataclass, replace

import numpy as np
from spod.sPOD_tools import give_interpolation_error, shifted_rPCA
from spod.transforms import transforms


@dataclass
class PODResult:
    """POD basis and amplitudes, truncated to n_modes."""

    modes: np.ndarray  # For training data
    time_amplitudes: np.ndarray  # For training data
    reconstruction: np.ndarray  # Truncated reconstruction of training data
    time_amplitudes_test: np.ndarray  # For test data


@dataclass
class sPODResult:
    """sPOD bases and amplitudes per frame (lists with one entry per frame)."""

    interpolation_error: float  # Interpolation error for the transformation
    offline_error: float  # Offline error for the sPOD reconstruction
    modes: list  # For training data
    time_amplitudes: list  # For training data
    reconstruction: np.ndarray  # Truncated reconstruction of training data
    time_amplitudes_test: list | None  # For test data (None when no test frames were given)
    ranks: list  # Modes kept per frame
    error_history: list  # shifted_rPCA's relative residual, one value per iteration


def pod(q: np.ndarray, q_test: np.ndarray, n_modes: int) -> PODResult:
    """Proper orthogonal decomposition of q, truncated to n_modes.

    With the thin SVD q = U S V^T, the modes are U[:, :n], the training amplitudes
    S[:n] V^T[:n] and the test amplitudes U[:, :n]^T q_test.

    Args:
        q: Training snapshots, shape (Nx, n_snapshots).
        q_test: Test snapshots, shape (Nx, n_test_snapshots).
        n_modes: Number of modes to keep. The paper uses 18 for POD-NN: as many
            outputs as sPOD-NN (2 frames x 8 modes + 2 shifts).
    """
    # Generate data for the POD-NN for comparison
    U, S, VT = np.linalg.svd(q, full_matrices=False)
    modes = U[:, :n_modes]
    time_amplitudes = np.diag(S[:n_modes]) @ VT[:n_modes, :]
    reconstruction = modes @ time_amplitudes

    time_amplitudes_test = modes.T @ q_test

    return PODResult(
        modes=modes,
        time_amplitudes=time_amplitudes,
        reconstruction=reconstruction,
        time_amplitudes_test=time_amplitudes_test,
    )


def orientation(amplitudes: np.ndarray) -> np.ndarray:
    """Sign of the largest-magnitude entry of each amplitude row (one sign per mode)."""
    largest = np.argmax(np.abs(amplitudes), axis=1)
    return np.sign(amplitudes[np.arange(amplitudes.shape[0]), largest])


def align_signs(result: PODResult, reference_orientation) -> PODResult:
    """Flip POD modes so their orientation matches a reference basis.

    A network trained on one basis predicts amplitudes in that basis' signs. Each
    mode whose orientation (see orientation()) differs from reference_orientation
    is negated, together with its training and test amplitudes; the product
    modes @ amplitudes, and so the reconstruction, is unchanged.

    Args:
        result: POD result to align.
        reference_orientation: One sign (+1 or -1) per mode, e.g. the orientation
            recorded for the paper's POD basis in tests/golden/synthetic.json.
    """
    flips = orientation(result.time_amplitudes) * np.asarray(reference_orientation)
    return replace(
        result,
        modes=result.modes * flips,
        time_amplitudes=flips[:, None] * result.time_amplitudes,
        time_amplitudes_test=flips[:, None] * result.time_amplitudes_test,
    )


def spod(
    q: np.ndarray,
    q_test: list | None,
    trafos: list,
    n_modes: int | None,
    n_iter: int,
    mu_factor: float = 0.005,
    lamda_factor: float = 10.0,
    penalty: str = "matrix",  # "matrix" | "grid"
    grid_shape: tuple | None = None,
    nmodes_max: int | None = None,
    eps: float = 1e-16,
) -> sPODResult:
    """Shifted POD of q with spod.shifted_rPCA, truncated to n_modes per frame.

    The penalty parameters follow one of the paper's two formulas, for q of shape (N, M):
    "matrix" (synthetic, 1D): mu0 = N M / (4 sum|q|) * mu_factor, lambda0 = lamda_factor / sqrt(max(N, M));
    "grid" (2D cases):        mu0 = N / (4 sum|q|) * mu_factor,   lambda0 = lamda_factor / sqrt(max(Nx, Ny)).
    For frame k with modes U_k and amplitudes A_k = diag(sigma_k) V_k^T, the truncated
    reconstruction is sum_k T_k(U_k A_k), and the test amplitudes are
    U_k^T q_test[k], projected from the unshifted test frames.

    Args:
        q: Training snapshots, shape (N, n_snapshots).
        q_test: Unshifted test frames [q1_test, q2_test, ...] (SyntheticData.frames),
            or None for no test amplitudes (e.g. when the run itself produces test frames).
        trafos: One spod transform per frame (SyntheticData.transforms).
        n_modes: Modes kept per frame (the paper uses 8 for synthetic); None keeps
            the rank sPOD settles on (the wildfire cases).
        n_iter: shifted_rPCA iterations (the paper uses 300 for synthetic).
        mu_factor, lamda_factor: Case-specific penalty factors (synthetic: 0.005, 10).
        penalty: Which formula sets mu0 and lambda0, "matrix" or "grid".
        grid_shape: The spatial grid (Nx, Ny), needed for penalty="grid".
        nmodes_max: Largest rank sPOD may use per frame; None means n_modes + 20
            (the paper's synthetic setting).
        eps: shifted_rPCA stops once its relative residual falls below eps.

    Returns:
        The bases, amplitudes and truncated reconstruction, the modes kept per frame,
        and three diagnostics: the transformation interpolation error (the lowest
        error sPOD can reach on this grid), the relative error of the truncated
        reconstruction, and shifted_rPCA's residual per iteration.
    """
    if penalty not in ("matrix", "grid"):
        raise ValueError(f"penalty must be 'matrix' or 'grid', got {penalty!r}")
    if penalty == "grid" and grid_shape is None:
        raise ValueError("penalty='grid' needs grid_shape")
    if n_modes is None and nmodes_max is None:
        raise ValueError("n_modes=None (keep sPOD's rank) needs an explicit nmodes_max")

    ##########################################
    # Calculate the transformation interpolation error
    interp_err = give_interpolation_error(q, trafos[0])

    qmat = q.copy()
    [N, M] = np.shape(qmat)

    # mu0 and lambd0 from `penalty`
    if penalty == "matrix":
        mu0 = N * M / (4 * np.sum(np.abs(qmat))) * mu_factor
        lambd0 = 1 / np.sqrt(np.maximum(M, N)) * lamda_factor
    else:
        mu0 = N / (4 * np.sum(np.abs(qmat))) * mu_factor
        lambd0 = 1 / np.sqrt(max(grid_shape)) * lamda_factor

    if nmodes_max is None:
        nmodes_max = n_modes + 20  # the paper's synthetic setting
    ret = shifted_rPCA(q, trafos, nmodes_max=nmodes_max, eps=eps, Niter=n_iter, use_rSVD=True, mu=mu0, lambd=lambd0)
    sPOD_frames = ret.frames

    ###########################################
    # Calculate the time amplitudes for training  and testing data
    # (with n_modes=None, [:None] keeps every mode: the rank sPOD settled on)
    U_list = []
    TA_list = []
    TA_test_list = []
    qtrunc = 0
    cnt = 0
    for frame in sPOD_frames:
        VT = frame.modal_system["VT"][:n_modes, :]
        S = frame.modal_system["sigma"][:n_modes]
        time_amplitude = np.diag(S) @ VT
        TA_list.append(time_amplitude)
        U_list.append(frame.modal_system["U"][:, :n_modes])
        qtrunc += trafos[cnt].apply(U_list[cnt] @ TA_list[cnt])

        if q_test is not None:
            TA_test_list.append(U_list[cnt].T @ q_test[cnt])

        cnt = cnt + 1

    err_trunc = np.linalg.norm(q - qtrunc) / np.linalg.norm(q)

    return sPODResult(
        interpolation_error=interp_err,
        offline_error=err_trunc,
        modes=U_list,
        time_amplitudes=TA_list,
        reconstruction=qtrunc,
        time_amplitudes_test=TA_test_list if q_test is not None else None,
        ranks=[U.shape[1] for U in U_list],
        error_history=list(ret.rel_err_hist),
    )


def reconstruct(modes: list, amplitudes: list, shifts: list, grid) -> np.ndarray:
    """Rebuild the field from per-frame modes, amplitudes and shifts.

    q = sum_k T_{shifts[k]} (modes[k] @ amplitudes[k]), the inverse of the sPOD.
    Used for sPOD-NN (amplitudes and shifts from the network) and sPOD-I
    (interpolated); shifts[k] has one value per snapshot (column).
    """
    q = 0
    for U, a, s in zip(modes, amplitudes, shifts, strict=True):
        trafo = transforms(
            [grid.Nx, 1, 1, a.shape[1]],
            [grid.L],
            shifts=np.ravel(s),
            dx=[grid.dx],
            use_scipy_transform=False,
            interp_order=5,
        )
        q = q + trafo.apply(U @ a)
    return q
