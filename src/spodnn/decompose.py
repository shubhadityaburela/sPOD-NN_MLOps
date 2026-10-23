"""Decomposition of one case, from its generated dataset to sPOD's results.

The module follows the order of the work:

1. Load: load_variable reads one variable (q for synthetic, T or S for wildfire)
   from data/<case>.zarr and splits it into training and test beta/mu, in the
   coordinates sPOD works in (x for synthetic and 1D, polar (r, theta) for 2D).
2. Transforms: build_transforms makes one spod transform per frame (FRAMES).
3. sPOD: run_spod decomposes one variable with spodnn.decomposition.spod (and,
   for wildfire, a second sPOD of the test beta that gives its frames), adds the
   POD baseline and the shift targets the network learns.
4. Per config file: decompose_case checks the config against the dataset and
   yields one result per variable (spodnn.decomposition_store writes them).

Ported from paper/wildfire*_sup.py (__init__, run_sPOD and test_data, the
notebooks' POD and shift cells) and spodnn.synthetic.generate.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from spod.transforms import transforms

from spodnn.dataset import open_case, read_manifest
from spodnn.decomposition import PODResult, pod, spod, sPODResult
from spodnn.decomposition_config import TransformConfig, sPODConfig
from spodnn.wildfire.polar import to_polar
from spodnn.wildfire.shifts import expand_shifts

# Frame types per layout, in frame order: the paper's choice per case.
FRAMES = {
    "synthetic": ["shift", "shift"],
    "1d": ["shift", "identity", "shift"],
    "2d": ["shift", "identity"],
    "2d_ww": ["shift", "identity"],
}


@dataclass(frozen=True)
class Variable:
    """One variable of one case: training and test snapshots plus what the transforms need.

    Snapshot matrices have one column per snapshot, the beta/mu one after another
    (columns k*Nt .. (k+1)*Nt - 1 belong to the k-th value of the split).
    """

    name: str  # "q" (synthetic), "T" or "S"
    layout: str  # a key of FRAMES
    q_train: np.ndarray  # (N, Nt * n_train), float64
    q_test: np.ndarray  # (N, Nt * n_test), float64
    shifts_train: np.ndarray  # as stored, β concatenated along the last (time) axis
    shifts_test: np.ndarray
    grid_shape: tuple  # (Nx, 1) or (N_r, N_theta)
    domain: list  # [L] or [r_max, theta_max]
    spacing: list  # [dx] or [dr, dtheta]
    frames_test: list | None = None  # synthetic only: analytic test frames [q1, q2]
    # 2D only: the same snapshots on the Cartesian grid (N = Nx*Ny, points in Fortran order),
    # for the POD baseline, which needs no polar map. None where q_train already is Cartesian.
    q_train_cartesian: np.ndarray | None = None
    q_test_cartesian: np.ndarray | None = None


def variable_names(layout: str) -> list[str]:
    """The variables a case is decomposed into, each with its own sPOD run."""
    return ["q"] if layout == "synthetic" else ["T", "S"]


def layout(data_config: dict) -> str:
    """The case's layout: "synthetic", "1d", "2d" or "2d_ww".

    Read from the data config the manifest stores (manifest["config"]): Ny == 1
    is the 1D case, edge_detect the 2D case with wind (2D_ww).
    """
    if data_config["kind"] == "synthetic":
        return data_config["kind"]
    else:
        if data_config["grid"]["Ny"] == 1:
            return "1d"
        elif data_config["edge_detect"]:
            return "2d_ww"
        else:
            return "2d"


def _rows(name, lay, nx):
    """Which rows of a β's snapshot matrix hold variable `name`, as spodnn generate wrote them."""
    if lay == "synthetic":
        return slice(None)  # one variable: all rows
    if lay == "1d":
        return slice(0, nx) if name == "T" else slice(nx, 2 * nx)  # stacked [T; S]
    return slice(0, None, 2) if name == "T" else slice(1, None, 2)  # interleaved per grid point


def _polar_matrix(rows, x, y, fill_value):
    """One β's 2D field (Nx*Ny, Nt) -> (q_polar (N_r*N_theta, Nt), r, theta)."""
    nt = rows.shape[-1]
    field = rows.reshape(len(x), len(y), nt, order="F")  # generate wrote the points in Fortran order
    polar, r, theta, _ = to_polar(field, x, y, fill_value)
    return polar.reshape(-1, nt), r, theta  # C order, as the paper


def _snapshots(group, indices, rows, lay, x, y):
    """Snapshot matrix (N, len(indices)*Nt) of one variable for the given β, in sPOD's coordinates.

    Returns (q, q_cartesian, r, theta): for the 2D layouts q is polar and q_cartesian,
    r and theta are set; otherwise q is already Cartesian and the other three are None.
    """
    # Orthogonal indexing reads only these β and only this variable's rows: (n_beta, N, Nt).
    block = group["snapshots"].oindex[indices, rows, :].astype(np.float64)
    # Put the β side by side as column blocks, like np.concatenate(..., axis=1).
    cartesian = block.transpose(1, 0, 2).reshape(block.shape[1], -1)
    if lay not in ("2d", "2d_ww"):
        return cartesian, None, None, None
    fill_value = 1 if lay == "2d_ww" else 0  # as in paper/wildfire2D*_sup.py
    mapped = [_polar_matrix(b, x, y, fill_value) for b in block]
    r, theta = mapped[0][1], mapped[0][2]  # the same polar grid for every β
    return np.concatenate([q for q, _, _ in mapped], axis=1), cartesian, r, theta


def load_variable(path, name: str) -> Variable:
    """Load variable `name` of the store at path, split into training and test β (or μ).

    The store's checksums are verified first. Snapshots are stored as float32 and
    returned as float64; the 2D cases are mapped to polar coordinates.
    """
    group = open_case(path)  # refuses a store whose checksums don't match
    manifest = dict(group.attrs)["manifest"]
    data_config = manifest["config"]
    lay = layout(data_config)
    if name not in variable_names(lay):
        raise ValueError(f"{path} has no variable {name!r}; it has {variable_names(lay)}")
    test = manifest["test_indices"]
    train = [i for i in range(group["params"].shape[0]) if i not in test]
    x, y = group["x"][:], group["y"][:]

    rows = _rows(name, lay, len(x))
    q_train, q_train_cartesian, r, theta = _snapshots(group, train, rows, lay, x, y)
    q_test, q_test_cartesian, _, _ = _snapshots(group, test, rows, lay, x, y)

    # Stored per β: synthetic (2, Nt), 1d (3, Nt), 2d (2, 2, Nt), 2d_ww (N_theta, Nt); time is last.
    shifts_train = np.concatenate([group["shifts"][i] for i in train], axis=-1)
    shifts_test = np.concatenate([group["shifts"][i] for i in test], axis=-1)

    if lay == "synthetic":
        grid_shape, domain, spacing = (len(x), 1), [data_config["grid"]["L"]], [x[1] - x[0]]  # x starts at -L/2
    elif lay == "1d":
        grid_shape, domain, spacing = (len(x), 1), [x[-1]], [x[1] - x[0]]  # paper: L = x[-1] (= Lx)
    else:
        grid_shape, domain, spacing = (len(x), len(y)), [r[-1], theta[-1]], [r[1] - r[0], theta[1] - theta[0]]

    frames_test = None
    if lay == "synthetic":
        frames = group["frames"].oindex[test].astype(np.float64)  # (n_test, 2, Nx, Nt)
        frames_test = [np.concatenate(list(frames[:, k]), axis=1) for k in range(frames.shape[1])]

    return Variable(
        name=name,
        layout=lay,
        q_train=q_train,
        q_test=q_test,
        shifts_train=shifts_train,
        shifts_test=shifts_test,
        grid_shape=grid_shape,
        domain=domain,
        spacing=spacing,
        frames_test=frames_test,
        q_train_cartesian=q_train_cartesian,
        q_test_cartesian=q_test_cartesian,
    )


def build_transforms(var: Variable, split: str, cfg: TransformConfig) -> list:
    """One spod transform per frame (FRAMES[var.layout]) for the "train" or "test" snapshots.

    The shifts are passed per frame in the form spod.transforms expects:
    synthetic and 1d one value per snapshot, (n,); 2d an (r, theta) pair per
    snapshot, (2, n); 2d_ww an (r, theta) pair per pixel and snapshot,
    (2, N_r*N_theta, n), rebuilt from the compact radial shift.
    """
    shifts = var.shifts_train if split == "train" else var.shifts_test
    n = shifts.shape[-1]
    data_shape = [*var.grid_shape, 1, n]

    if var.layout == "2d_ww":
        # Frame 0: the radial shift of every pixel (expanded only here, for as long as the
        # shift matrices take to build). Frame 1 is the identity, which in 2D ignores its shifts.
        shifts_input = [expand_shifts(shifts, var.grid_shape[0])[0].reshape(2, -1, n), None]
    else:
        # synthetic, 1d: shifts[k] has shape (n,); 2d: (2, n). A 1D identity frame still
        # needs its (zero) shifts, because sPOD builds its matrices regardless.
        shifts_input = list(shifts)

    return [
        transforms(
            data_shape,
            var.domain,
            trafo_type=trafo_type,
            shifts=shifts_input[k],
            dx=var.spacing,
            use_scipy_transform=cfg.use_scipy_transform,
            interp_order=cfg.interp_order,
        )
        for k, trafo_type in enumerate(FRAMES[var.layout])
    ]


@dataclass
class VariableDecomposition:
    """sPOD of one variable, its POD baseline and the shift targets the network learns.

    The network (Phase 3) maps (t, beta) to the sPOD amplitudes of every frame
    followed by the shift targets; POD-NN maps it to as many POD amplitudes.
    """

    name: str  # "q", "T" or "S"
    spod: sPODResult  # modes, amplitudes (train and test), ranks, errors
    test_spod: sPODResult | None  # wildfire: the sPOD of the test beta that gave its frames; synthetic: None
    pod: PODResult  # Cartesian baseline with sum(ranks) + n_shift_outputs modes
    shifts_train: np.ndarray  # (n_shift_outputs, M_train)
    shifts_test: np.ndarray  # (n_shift_outputs, M_test)
    shift_basis: np.ndarray | None  # 2D_ww: modes of the shift POD, (N_theta, shift_modes); else None


def _shift_targets(var: Variable, shift_modes: int | None):
    """(train, test, basis): the shifts the network predicts, as in the paper's notebooks.

    synthetic: both frames' shifts; 1D: frames 1 and 3 (frame 2 is the identity);
    2D: the radial shift of frame 1 (its angular shift is 0); 2D_ww: the amplitudes
    of a rank-`shift_modes` POD of the radial shifts. The test amplitudes are
    projected onto the training shift basis, so they are in the network's coordinates.
    """
    if var.layout == "2d_ww":
        result = pod(var.shifts_train, var.shifts_test, shift_modes)
        return result.time_amplitudes, result.time_amplitudes_test, result.modes
    if var.layout == "1d":
        return var.shifts_train[[0, 2]], var.shifts_test[[0, 2]], None
    if var.layout == "2d":
        return var.shifts_train[0, :1], var.shifts_test[0, :1], None
    return var.shifts_train, var.shifts_test, None


def run_spod(var: Variable, cfg, spod_cfg: sPODConfig) -> VariableDecomposition:
    """Decompose one variable: sPOD, test amplitudes, shift targets and the POD baseline.

    Args:
        var: The variable, from load_variable.
        cfg: The case's decomposition config (penalty, transform, shift_modes).
        spod_cfg: This variable's sPOD settings (cfg.spod, or cfg.spod.T / cfg.spod.S).

    The test frames are the analytic ones for synthetic; for wildfire they come
    from a second sPOD of the test beta (all the modes it finds, as the paper's
    build_field()). It runs first, so its shift matrices are freed before the
    larger training run builds its own.
    """
    settings = dict(
        n_iter=spod_cfg.n_iter,
        mu_factor=spod_cfg.mu_factor,
        lamda_factor=spod_cfg.lambda_factor,
        penalty=cfg.penalty,
        grid_shape=var.grid_shape,
        nmodes_max=spod_cfg.nmodes_max,
        eps=spod_cfg.eps,
    )

    if var.frames_test is not None:
        frames_test, test_spod = var.frames_test, None
    else:
        trafos_test = build_transforms(var, "test", cfg.transform)
        test_spod = spod(var.q_test, None, trafos_test, n_modes=None, **settings)
        frames_test = [U @ A for U, A in zip(test_spod.modes, test_spod.time_amplitudes, strict=True)]
        del trafos_test

    trafos_train = build_transforms(var, "train", cfg.transform)
    result = spod(var.q_train, frames_test, trafos_train, n_modes=spod_cfg.nmodes, **settings)
    del trafos_train

    shifts_train, shifts_test, shift_basis = _shift_targets(var, cfg.shift_modes)

    # POD-NN gets as many outputs as sPOD-NN, on the Cartesian snapshots (no polar map).
    n_outputs = sum(result.ranks) + len(shifts_train)
    q_train = var.q_train if var.q_train_cartesian is None else var.q_train_cartesian
    q_test = var.q_test if var.q_test_cartesian is None else var.q_test_cartesian
    pod_result = pod(q_train, q_test, n_outputs)

    return VariableDecomposition(
        name=var.name,
        spod=result,
        test_spod=test_spod,
        pod=pod_result,
        shifts_train=shifts_train,
        shifts_test=shifts_test,
        shift_basis=shift_basis,
    )


def check_config(cfg, lay: str) -> None:
    """Raise ValueError if the decomposition config doesn't fit a dataset of this layout.

    These are the checks a config file can't make on its own (spodnn.decomposition_config):
    the kind must match the data, scipy shifts exist only for 2D grids, and the
    shift POD (shift_modes) belongs to 2D_ww alone.
    """
    data_kind = "synthetic" if lay == "synthetic" else "wildfire"
    if cfg.kind != data_kind:
        raise ValueError(f"a {cfg.kind} decomposition config can't decompose {data_kind} data")
    if cfg.transform.use_scipy_transform and lay not in ("2d", "2d_ww"):
        raise ValueError(f"use_scipy_transform only works on 2D grids, but this dataset is {lay}")
    if (cfg.shift_modes is not None) != (lay == "2d_ww"):
        raise ValueError(f"shift_modes must be set for 2d_ww data and only there (dataset: {lay})")


def decompose_case(cfg, data_path) -> Iterator[tuple[Variable, VariableDecomposition]]:
    """Decompose every variable of the dataset at data_path, one after another.

    Yields (variable, result) per variable, so only one variable's snapshots and
    results are in memory at a time (like generate_case yields one beta at a time).
    """
    lay = layout(read_manifest(data_path)["config"])
    check_config(cfg, lay)
    for name in variable_names(lay):
        var = load_variable(data_path, name)
        spod_cfg = cfg.spod if lay == "synthetic" else getattr(cfg.spod, name)
        yield var, run_spod(var, cfg, spod_cfg)
