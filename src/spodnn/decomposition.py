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

from spod.sPOD_tools import give_interpolation_error, shifted_rPCA
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class PODResult:
    """POD basis and amplitudes, truncated to n_modes."""

    modes: np.ndarray # For training data
    time_amplitudes: np.ndarray # For training data
    reconstruction: np.ndarray # Truncated reconstruction of training data
    time_amplitudes_test: np.ndarray # For test data


@dataclass
class sPODResult:
    """sPOD bases and amplitudes per frame (lists with one entry per frame)."""

    interpolation_error: float # Interpolation error for the transformation
    offline_error: float # Offline error for the sPOD reconstruction
    modes: list # For training data
    time_amplitudes: list # For training data
    reconstruction: np.ndarray # Truncated reconstruction of training data
    time_amplitudes_test: list # For test data


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

    return PODResult(modes=modes, time_amplitudes=time_amplitudes, reconstruction=reconstruction, time_amplitudes_test=time_amplitudes_test)


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


def spod(q: np.ndarray, q_test: list, transforms: list, n_modes: int, n_iter: int, mu_factor: float = 0.005, lamda_factor: float = 10.0) -> sPODResult:
    """Shifted POD of q with spod.shifted_rPCA, truncated to n_modes per frame.

    The penalty parameters follow the paper: mu0 = N M / (4 sum|q|) * mu_factor
    and lambda0 = lambda_factor / sqrt(max(N, M)) for q of shape (N, M). For frame
    k with modes U_k and amplitudes A_k = diag(sigma_k) V_k^T, the truncated
    reconstruction is sum_k T_k(U_k A_k), and the test amplitudes are
    U_k^T q_test[k], projected from the unshifted test frames.

    Args:
        q: Training snapshots, shape (Nx, n_snapshots).
        q_test: Unshifted test frames [q1_test, q2_test] (SyntheticData.frames).
        transforms: One spod transform per frame (SyntheticData.transforms).
        n_modes: Modes kept per frame (the paper uses 8).
        n_iter: shifted_rPCA iterations (the paper uses 300).
        mu_factor, lamda_factor: Case-specific penalty factors (synthetic: 0.005, 10).

    Returns:
        The bases, amplitudes and truncated reconstruction, plus two diagnostics:
        the transformation interpolation error (the lowest error sPOD can reach on
        this grid) and the relative error of the truncated reconstruction.
    """
    ##########################################
    # Calculate the transformation interpolation error
    interp_err = give_interpolation_error(q, transforms[0])

    # Calculate the time amplitudes
    qmat = q.copy()
    [N, M] = np.shape(qmat)
    mu0 = N * M / (4 * np.sum(np.abs(qmat))) * mu_factor
    lambd0 = 1 / np.sqrt(np.maximum(M, N)) * lamda_factor
    ret = shifted_rPCA(q, transforms, nmodes_max=n_modes + 20, eps=1e-16, Niter=n_iter,
                        use_rSVD=True, mu=mu0, lambd=lambd0)
    sPOD_frames, _, _ = ret.frames, ret.data_approx, ret.rel_err_hist

    ###########################################
    # Calculate the time amplitudes for training  and testing data
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
        qtrunc += transforms[cnt].apply(U_list[cnt] @ TA_list[cnt])

        TA_test_list.append(U_list[cnt].T @ q_test[cnt])

        cnt = cnt + 1

    err_trunc = np.linalg.norm(q - qtrunc) / np.linalg.norm(q)

    return sPODResult(interpolation_error=interp_err, offline_error=err_trunc, modes=U_list, time_amplitudes=TA_list, reconstruction=qtrunc, time_amplitudes_test=TA_test_list)
