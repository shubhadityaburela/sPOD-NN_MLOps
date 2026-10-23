"""Tests for spodnn.decompose: loading one variable from a store, and building its transforms.

The stores are tiny and written here: every snapshot value encodes where it came
from (beta index, variable, grid point, time step), so a test can check that each
row and column ended up exactly where it belongs.
"""

from dataclasses import replace

import numpy as np
import pytest
import zarr
from conftest import CONFIGS_DIR
from numpy.testing import assert_allclose, assert_array_equal

import spodnn.decompose as decompose
from spodnn.config import load_config
from spodnn.dataset import _array_sha256
from spodnn.decompose import FRAMES, Variable, build_transforms, layout, load_variable, run_spod, variable_names
from spodnn.decomposition_config import (
    SyntheticDecompositionConfig,
    TransformConfig,
    WildfireDecompositionConfig,
    WildfireSPODConfig,
    sPODConfig,
)
from spodnn.synthetic import Grid, generate

NX, NY, NT = 6, 4, 3  # small enough that every value below stays exact in float32
TEST = [1]  # a test beta in the middle: the split must follow the indices, not the order
N_BETA = 4
TRAIN = [0, 2, 3]
VAR = {"q": 0, "T": 0, "S": 1}

DATA_CONFIGS = {
    "synthetic": {"kind": "synthetic", "grid": {"L": 2.0}},
    "1d": {"kind": "wildfire", "grid": {"Ny": 1}, "edge_detect": False},
    "2d": {"kind": "wildfire", "grid": {"Ny": NY}, "edge_detect": False},
    "2d_ww": {"kind": "wildfire", "grid": {"Ny": NY}, "edge_detect": True},
}


def code(beta, var, point, step):
    """The value stored for (beta index, variable, grid point, time step)."""
    return beta * 100_000 + var * 10_000 + point * 10 + step


def snapshot(lay, beta):
    """One beta's snapshot matrix in the layout spodnn generate writes."""
    t = np.arange(NT)
    if lay == "synthetic":
        return np.array([[code(beta, 0, p, j) for j in t] for p in range(NX)])
    if lay == "1d":  # stacked [T; S]
        return np.array([[code(beta, v, p, j) for j in t] for v in (0, 1) for p in range(NX)])
    # 2D: grid points in Fortran order (p = ix + NX * iy), T and S interleaved per point
    return np.array([[code(beta, v, p, j) for j in t] for p in range(NX * NY) for v in (0, 1)])


def shifts(lay, beta):
    """One beta's shifts as stored; the value encodes (beta, frame, time step)."""
    shape = {"synthetic": (2, NT), "1d": (3, NT), "2d": (2, 2, NT), "2d_ww": (NY, NT)}[lay]
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return beta * 1000 + values


def write_store(path, lay, snapshots=None):
    """A store with the arrays and manifest load_variable reads (checksums included)."""
    group = zarr.open_group(store=str(path), mode="w")
    snaps = snapshots if snapshots is not None else np.stack([snapshot(lay, i) for i in range(N_BETA)])
    group.create_array("snapshots", data=snaps.astype(np.float32))
    group.create_array("shifts", data=np.stack([shifts(lay, i) for i in range(N_BETA)]))
    group.create_array("params", data=np.arange(N_BETA, dtype=np.float64))
    x = np.arange(-NX // 2, NX // 2) / NX * 2.0 if lay == "synthetic" else np.arange(1, NX + 1, dtype=np.float64)
    group.create_array("x", data=x)
    group.create_array("y", data=np.arange(1, NY + 1, dtype=np.float64) if lay.startswith("2d") else np.zeros(1))
    if lay == "synthetic":
        frames = np.stack([[snapshot(lay, i) + 0.5 * k for k in (0, 1)] for i in range(N_BETA)])
        group.create_array("frames", data=frames.astype(np.float32))
    checksums = {name: _array_sha256(group[name]) for name in group.array_keys()}
    group.attrs["manifest"] = {"config": DATA_CONFIGS[lay], "test_indices": TEST, "checksums": checksums}
    return path


@pytest.fixture
def no_polar(monkeypatch):
    """Replace the polar map by the identity, so the Cartesian field stays readable; records fill values."""
    fills = []

    def fake_to_polar(field, x, y, fill_value):
        fills.append(fill_value)
        return field.copy(), np.linspace(0, 5, len(x)), np.linspace(-1, 1, len(y)), None

    monkeypatch.setattr(decompose, "to_polar", fake_to_polar)
    return fills


# ---------------------------------------------------------------- layout and names


@pytest.mark.parametrize(
    ("case", "expected"),
    [("synthetic", "synthetic"), ("wildfire1d", "1d"), ("wildfire2d", "2d"), ("wildfire2d_ww", "2d_ww")],
)
def test_layout_of_every_case(case, expected):
    # The data config as the manifest stores it.
    assert layout(load_config(CONFIGS_DIR / f"{case}.yaml").model_dump(mode="json")) == expected


def test_variable_names():
    assert variable_names("synthetic") == ["q"]
    assert all(variable_names(lay) == ["T", "S"] for lay in ("1d", "2d", "2d_ww"))


# ---------------------------------------------------------------- load_variable


@pytest.mark.parametrize(("lay", "name"), [("synthetic", "q"), ("1d", "T"), ("1d", "S")])
def test_1d_rows_and_split(tmp_path, lay, name):
    var = load_variable(write_store(tmp_path / "s.zarr", lay), name)
    expected = lambda betas: np.concatenate(  # noqa: E731
        [[[code(i, VAR[name], p, j) for j in range(NT)] for p in range(NX)] for i in betas], axis=1
    )
    assert var.q_train.dtype == np.float64
    assert_array_equal(var.q_train, expected(TRAIN))  # beta blocks side by side, in index order
    assert_array_equal(var.q_test, expected(TEST))
    assert var.q_train_cartesian is None and var.q_test_cartesian is None  # q_train already is Cartesian


@pytest.mark.parametrize("name", ["T", "S"])
@pytest.mark.parametrize("lay", ["2d", "2d_ww"])
def test_2d_rows_are_deinterleaved_in_fortran_order(tmp_path, no_polar, lay, name):
    var = load_variable(write_store(tmp_path / "s.zarr", lay), name)
    assert var.q_train.shape == (NX * NY, len(TRAIN) * NT)
    for block, i in enumerate(TRAIN):
        field = var.q_train[:, block * NT : (block + 1) * NT].reshape(NX, NY, NT)  # the (fake) polar field
        for ix, iy, j in np.ndindex(NX, NY, NT):
            assert field[ix, iy, j] == code(i, VAR[name], ix + NX * iy, j)
    assert set(no_polar) == {1 if lay == "2d_ww" else 0}  # the paper's fill values
    # The Cartesian copy for the POD baseline: the variable's rows as stored, β side by side.
    expected = lambda betas: np.concatenate(  # noqa: E731
        [[[code(i, VAR[name], p, j) for j in range(NT)] for p in range(NX * NY)] for i in betas], axis=1
    )
    assert_array_equal(var.q_train_cartesian, expected(TRAIN))
    assert_array_equal(var.q_test_cartesian, expected(TEST))


@pytest.mark.parametrize("lay", ["synthetic", "1d", "2d", "2d_ww"])
def test_shifts_are_concatenated_along_time(tmp_path, no_polar, lay):
    var = load_variable(write_store(tmp_path / "s.zarr", lay), variable_names(lay)[0])
    assert_array_equal(var.shifts_train, np.concatenate([shifts(lay, i) for i in TRAIN], axis=-1))
    assert_array_equal(var.shifts_test, np.concatenate([shifts(lay, i) for i in TEST], axis=-1))


def test_grid_per_layout(tmp_path, no_polar):
    synthetic = load_variable(write_store(tmp_path / "a.zarr", "synthetic"), "q")
    assert (synthetic.grid_shape, synthetic.domain) == ((NX, 1), [2.0])  # L from the config: x starts at -L/2
    assert_allclose(synthetic.spacing, [2.0 / NX])

    fire_1d = load_variable(write_store(tmp_path / "b.zarr", "1d"), "T")
    assert (fire_1d.grid_shape, fire_1d.domain, fire_1d.spacing) == ((NX, 1), [NX], [1.0])  # L = x[-1]

    fire_2d = load_variable(write_store(tmp_path / "c.zarr", "2d"), "T")
    assert fire_2d.grid_shape == (NX, NY)
    assert_allclose(fire_2d.domain, [5, 1])  # [r[-1], theta[-1]] of the (fake) polar grid
    assert_allclose(fire_2d.spacing, [1, 2 / (NY - 1)])


def test_only_synthetic_has_test_frames(tmp_path):
    var = load_variable(write_store(tmp_path / "s.zarr", "synthetic"), "q")
    for k, frame in enumerate(var.frames_test):
        assert frame.dtype == np.float64
        assert_array_equal(frame, var.q_test + 0.5 * k)
    assert load_variable(write_store(tmp_path / "w.zarr", "1d"), "T").frames_test is None


def test_polar_matrix_fills_points_outside_the_image():
    # The polar grid reaches the corners' radius, so along the axes its outer points lie
    # outside the image and take the fill value (load_variable: 0 for 2D, 1 for 2D_ww).
    n = 20  # on much smaller grids the spline interpolation never lands exactly on the fill value
    x = np.arange(1, n + 1, dtype=np.float64)
    for fill_value in (0, 1):
        q, r, theta = decompose._polar_matrix(np.zeros((n * n, 2)), x, x, fill_value)
        assert q.shape == (n * n, 2)
        assert (q == fill_value).any() and set(np.unique(q)) <= {0.0, float(fill_value)}
    assert (len(r), len(theta)) == (n, n)


def test_unknown_variable_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="has no variable 'T'"):
        load_variable(write_store(tmp_path / "s.zarr", "synthetic"), "T")


def test_corrupted_store_is_refused(tmp_path):
    path = write_store(tmp_path / "s.zarr", "1d")
    zarr.open_group(store=str(path), mode="r+")["snapshots"][0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="checksum mismatch in snapshots"):
        load_variable(path, "T")


# ---------------------------------------------------------------- build_transforms

OWN = TransformConfig(use_scipy_transform=False, interp_order=5)
SCIPY = TransformConfig(use_scipy_transform=True, interp_order=3)


def variable(lay, shifts_train, n_points, grid_shape, domain, spacing):
    q = np.zeros((n_points, shifts_train.shape[-1]))
    return Variable(
        name="T",
        layout=lay,
        q_train=q,
        q_test=q[:, :1],
        shifts_train=shifts_train,
        shifts_test=shifts_train[..., :1],
        grid_shape=grid_shape,
        domain=domain,
        spacing=spacing,
    )


def field(n_points, n, seed=0):
    return np.random.default_rng(seed).standard_normal((n_points, n))


@pytest.mark.parametrize(("lay", "n_frames"), [("synthetic", 2), ("1d", 3)])
def test_1d_frames_and_zero_shift(lay, n_frames):
    var = variable(lay, np.zeros((n_frames, 5)), 16, (16, 1), [16.0], [1.0])
    trafos = build_transforms(var, "train", OWN)
    assert [t.trafo_type for t in trafos] == FRAMES[lay]
    q = field(16, 5)
    for trafo in trafos:
        assert_allclose(trafo.apply(q), q, atol=1e-12)  # no shift, no change


def test_1d_shift_by_whole_cells_moves_the_field():
    cells = np.array([0, 1, 2, 3])
    var = variable("1d", np.stack([cells, 0 * cells, -cells]).astype(float), 16, (16, 1), [16.0], [1.0])
    frame_1, identity, frame_3 = build_transforms(var, "train", OWN)
    q = field(16, 4)
    for j, c in enumerate(cells):  # sPOD's convention: apply with shift s rolls the field by -s/dx cells
        assert_allclose(frame_1.apply(q)[:, j], np.roll(q[:, j], -c), atol=1e-12)
        assert_allclose(frame_3.apply(q)[:, j], np.roll(q[:, j], c), atol=1e-12)
    assert_array_equal(identity.apply(q), q)


def test_test_split_uses_the_test_shifts():
    var = variable("1d", np.zeros((3, 5)), 16, (16, 1), [16.0], [1.0])
    assert all(t.Ntime == 1 for t in build_transforms(var, "test", OWN))
    assert all(t.Ntime == 5 for t in build_transforms(var, "train", OWN))


def test_2d_frames_and_inverse():
    # (r, theta) shift per snapshot; scipy's shift is periodic, so reverse undoes apply.
    r_shift = np.array([0.0, 1.0, 2.5])
    var = variable("2d", np.stack([[r_shift, 0 * r_shift], np.zeros((2, 3))]), 64, (8, 8), [7.0, 7.0], [1.0, 1.0])
    frame_1, identity = build_transforms(var, "train", SCIPY)
    assert [frame_1.trafo_type, identity.trafo_type] == FRAMES["2d"]
    q = field(64, 3)
    assert_allclose(frame_1.apply(q)[:, 0], q[:, 0], atol=1e-12)  # zero shift
    assert_allclose(frame_1.reverse(frame_1.apply(q))[:, 1], q[:, 1], atol=1e-10)  # whole cell: invertible
    assert_array_equal(identity.apply(q), q)


@pytest.mark.slow  # numba compiles sPOD's general 2D shift (~7 s)
def test_2d_ww_expands_the_radial_shift():
    n_r = n_theta = 12  # sPOD's general 2D shift needs a square grid
    radial = np.zeros((n_theta, 2))
    radial[:, 1] = 2.0  # snapshot 1: every angle shifted by two radial cells
    var = variable("2d_ww", radial, n_r * n_theta, (n_r, n_theta), [n_r - 1.0, n_theta - 1.0], [1.0, 1.0])
    frame_1, identity = build_transforms(var, "train", OWN)
    assert [frame_1.trafo_type, identity.trafo_type] == FRAMES["2d_ww"]
    q = field(n_r * n_theta, 2)
    shifted = frame_1.apply(q)
    assert_allclose(shifted[:, 0], q[:, 0], atol=1e-12)  # zero shift
    # Along r only (axis 0), the same for every theta; -s/dx cells as in 1D.
    rolled = np.roll(q[:, 1].reshape(n_r, n_theta), -2, axis=0).ravel()
    assert_allclose(shifted[:, 1], rolled, atol=1e-12)
    assert_array_equal(identity.apply(q), q)


# ---------------------------------------------------------------- run_spod

SPOD = sPODConfig(n_iter=5, mu_factor=0.05, lambda_factor=10.0, nmodes=None, nmodes_max=4, eps=1e-16)


@pytest.fixture(scope="module")
def two_waves():
    """Small synthetic data (two structures moving apart): three training mu, one test mu."""
    grid = Grid(Nx=64, Nt=32)
    return grid, generate(np.array([0.1, 0.2, 0.3]), grid), generate(np.array([0.25]), grid)


def wave_variable(two_waves, lay):
    """The synthetic data as a "synthetic" variable (analytic test frames), or as a "1d" one:
    frames shift, identity, shift, with nothing in the identity frame and no test frames."""
    grid, train, test = two_waves
    s_train, s_test = np.concatenate(train.shifts), np.concatenate(test.shifts)  # (2, M)
    if lay == "1d":
        s_train = np.stack([s_train[0], 0 * s_train[0], s_train[1]])
        s_test = np.stack([s_test[0], 0 * s_test[0], s_test[1]])
    return Variable(
        name="q" if lay == "synthetic" else "T",
        layout=lay,
        q_train=train.q,
        q_test=test.q,
        shifts_train=s_train,
        shifts_test=s_test,
        grid_shape=(grid.Nx, 1),
        domain=[grid.L],
        spacing=[grid.dx],
        frames_test=test.frames if lay == "synthetic" else None,
    )


def config(lay):
    if lay == "synthetic":
        return SyntheticDecompositionConfig(penalty="matrix", transform=OWN, spod=SPOD)
    return WildfireDecompositionConfig(penalty="matrix", transform=OWN, spod=WildfireSPODConfig(T=SPOD, S=SPOD))


def test_run_spod_synthetic_projects_the_analytic_test_frames(two_waves):
    var = wave_variable(two_waves, "synthetic")
    result = run_spod(var, config("synthetic"), SPOD)
    assert result.test_spod is None
    for U, A_test, frame in zip(result.spod.modes, result.spod.time_amplitudes_test, var.frames_test, strict=True):
        assert_allclose(A_test, U.T @ frame, rtol=1e-12)
    assert_array_equal(result.shifts_train, var.shifts_train)  # both frames move: two shift outputs
    assert result.shift_basis is None
    assert result.pod.modes.shape[1] == sum(result.spod.ranks) + 2  # POD-NN: as many outputs as sPOD-NN


def test_run_spod_wildfire_gets_test_frames_from_a_second_spod(two_waves):
    var = wave_variable(two_waves, "1d")
    result = run_spod(var, config("1d"), SPOD)
    assert len(result.spod.modes) == len(result.test_spod.modes) == 3
    assert result.test_spod.time_amplitudes_test is None
    test_frames = [U @ A for U, A in zip(result.test_spod.modes, result.test_spod.time_amplitudes, strict=True)]
    for U, A_test, frame in zip(result.spod.modes, result.spod.time_amplitudes_test, test_frames, strict=True):
        assert A_test.shape == (U.shape[1], var.q_test.shape[1])
        assert_allclose(A_test, U.T @ frame, rtol=1e-12)
    assert_array_equal(result.shifts_train, var.shifts_train[[0, 2]])  # frame 2 is the identity
    assert_array_equal(result.shifts_test, var.shifts_test[[0, 2]])
    assert result.pod.modes.shape[1] == sum(result.spod.ranks) + 2


def test_pod_baseline_uses_the_cartesian_snapshots_when_given(two_waves):
    var = wave_variable(two_waves, "synthetic")
    rng = np.random.default_rng(0)
    cartesian = replace(
        var,
        q_train_cartesian=rng.standard_normal((40, var.q_train.shape[1])),
        q_test_cartesian=rng.standard_normal((40, var.q_test.shape[1])),
    )
    result = run_spod(cartesian, config("synthetic"), SPOD)
    assert result.pod.modes.shape[0] == 40  # the POD basis lives on the Cartesian grid
    assert result.spod.modes[0].shape[0] == var.q_train.shape[0]  # sPOD on q_train as before


def test_2d_shift_target_is_the_radial_shift_of_frame_1():
    shifts = np.random.default_rng(0).standard_normal((2, 2, 6))  # (frame, (r, theta), M)
    var = variable("2d", shifts, 4, (2, 2), [1.0, 1.0], [1.0, 1.0])
    train, test, basis = decompose._shift_targets(var, None)
    assert_array_equal(train, shifts[0, :1])
    assert_array_equal(test, shifts[0, :1, :1])
    assert basis is None


def test_2d_ww_shift_targets_use_the_training_shift_basis():
    rng = np.random.default_rng(0)
    basis_true = rng.standard_normal((10, 3))  # radial shifts of rank 3 over N_theta = 10 angles
    radial_train, radial_test = basis_true @ rng.standard_normal((3, 12)), basis_true @ rng.standard_normal((3, 4))
    var = replace(variable("2d_ww", radial_train, 4, (2, 2), [1.0, 1.0], [1.0, 1.0]), shifts_test=radial_test)
    train, test, basis = decompose._shift_targets(var, 3)
    assert (train.shape, test.shape, basis.shape) == ((3, 12), (3, 4), (10, 3))
    assert_allclose(basis @ train, radial_train, atol=1e-12)  # rank 3 is exact for rank-3 shifts
    assert_allclose(test, basis.T @ radial_test, rtol=1e-12)  # projected, not a basis of its own
    assert_allclose(basis @ test, radial_test, atol=1e-12)
