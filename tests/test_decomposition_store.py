"""Tests for spodnn.decomposition_store and `spodnn decompose`: the store, its manifest, skipping and atomic writes.

The datasets are tiny (written by spodnn generate's own code) and sPOD runs a
few iterations, so every test takes a fraction of a second.
"""

import logging
import shutil

import numpy as np
import pytest
import yaml
import zarr
from conftest import CONFIGS_DIR, small_wildfire
from pydantic import TypeAdapter

import spodnn.decomposition_store as store
from spodnn.cli import main
from spodnn.config import CaseConfig
from spodnn.dataset import open_case, read_manifest, verify, write_case
from spodnn.decompose import check_config
from spodnn.decomposition_config import DecompositionConfig, load_decomposition_config
from spodnn.decomposition_store import fingerprint, stale_reason, write_decomposition

FEW_ITERATIONS = {"n_iter": 3, "mu_factor": 0.05}


def decomposition_config(case, **spod):
    """The case's real decomposition config with a quick sPOD (few iterations)."""
    data = yaml.safe_load((CONFIGS_DIR / "decomposition" / f"{case}.yaml").read_text())
    sections = [data["spod"]] if data["kind"] == "synthetic" else [data["spod"]["T"], data["spod"]["S"]]
    for section in sections:
        section.update(FEW_ITERATIONS, **spod)
    return TypeAdapter(DecompositionConfig).validate_python(data)


def small_synthetic(**mu):
    data = yaml.safe_load((CONFIGS_DIR / "synthetic.yaml").read_text())
    data["grid"].update(Nx=64, Nt=32)
    data["mu"].update(mu)
    return TypeAdapter(CaseConfig).validate_python(data)


@pytest.fixture(scope="module")
def cfg():
    return decomposition_config("synthetic", nmodes=2, nmodes_max=4)


@pytest.fixture(scope="module")
def computed_once(cfg, tmp_path_factory):
    """A dataset and its decomposition, computed once for the module."""
    root = tmp_path_factory.mktemp("computed")
    write_case(small_synthetic(), "synthetic", root / "data")
    write_decomposition(cfg, "synthetic", root / "data", root / "out")
    return root


@pytest.fixture
def data_dir(computed_once, tmp_path):
    # Each test gets its own copies, so it may change or break them.
    shutil.copytree(computed_once / "data", tmp_path / "data")
    return tmp_path / "data"


@pytest.fixture
def decomposed(computed_once, data_dir, tmp_path):
    shutil.copytree(computed_once / "out", tmp_path / "out")
    return tmp_path / "out" / "synthetic.zarr"


# ---------------------------------------------------------------- what is stored


def test_store_holds_the_decomposition(cfg, data_dir, decomposed):
    group = open_case(decomposed)  # verifies every checksum
    manifest = read_manifest(decomposed)
    m_train, m_test = 3 * 32 + 2 * 32, 32  # five training mu, one test mu, Nt = 32
    for k in (0, 1):
        assert group[f"q.modes.{k}"].shape == (64, 2)
        assert group[f"q.amplitudes.{k}"].shape == (2, m_train)
        assert group[f"q.amplitudes_test.{k}"].shape == (2, m_test)
    assert group["q.shifts"].shape == (2, m_train)
    assert group["q.pod.modes"].shape == (64, 2 + 2 + 2)  # sum of ranks + two shift outputs
    assert "q.shift_basis" not in group

    assert manifest["schema"] == store.SCHEMA
    assert manifest["fingerprint"] == fingerprint(cfg, data_dir / "synthetic.zarr")
    q = manifest["variables"]["q"]
    assert (q["layout"], q["frames"], q["ranks"], q["grid_shape"]) == ("synthetic", ["shift", "shift"], [2, 2], [64, 1])
    assert len(q["error_history"]) == 3 and q["test_spod"] is None
    assert set(manifest["checksums"]) == set(group.array_keys())


@pytest.mark.slow  # ~5 s: two sPOD runs per variable on a real (small) 1D fire
def test_wildfire_store_has_both_variables_and_the_test_spod(tmp_path):
    data_dir = tmp_path / "data"
    write_case(small_wildfire("wildfire1d", Lx=200, Nx=400, Nt=400), "wildfire1d", data_dir)
    path = write_decomposition(decomposition_config("wildfire1d"), "wildfire1d", data_dir, tmp_path / "out")
    group, variables = open_case(path), read_manifest(path)["variables"]
    for v in ("T", "S"):
        assert [f"{v}.modes.{k}" in group for k in range(3)] == [True] * 3  # shift, identity, shift
        assert group[f"{v}.shifts"].shape == (2, 5 * 40)  # frames 1 and 3; 5 training beta x 40 snapshots
        assert variables[v]["test_spod"]["ranks"]  # the test frames came from a second sPOD
        assert group[f"{v}.pod.modes"].shape[1] == sum(variables[v]["ranks"]) + 2


# ---------------------------------------------------------------- skipping unchanged work


def _fail_if_called(cfg, data_path):
    raise AssertionError("decompose_case should not run")


def test_up_to_date_store_is_skipped(cfg, data_dir, decomposed, monkeypatch, caplog):
    monkeypatch.setattr(store, "decompose_case", _fail_if_called)
    with caplog.at_level(logging.INFO):
        assert write_decomposition(cfg, "synthetic", data_dir, decomposed.parent) == decomposed
    assert "up to date" in caplog.text


def test_changes_trigger_recomputation(cfg, data_dir, decomposed, monkeypatch):
    data = data_dir / "synthetic.zarr"
    assert stale_reason(decomposed, decomposition_config("synthetic", nmodes=1, nmodes_max=4), data) == "config changed"
    with monkeypatch.context() as mp:
        mp.setattr(store, "code_hash", lambda: "0" * 64)
        assert stale_reason(decomposed, cfg, data) == "decomposing code changed"
    with monkeypatch.context() as mp:
        mp.setattr(store, "version", lambda lib: "0.0")
        assert "numpy" in stale_reason(decomposed, cfg, data)
    write_case(small_synthetic(test=[0.2]), "synthetic", data_dir)  # regenerated with other content
    assert stale_reason(decomposed, cfg, data) == "input data changed"


def test_corruption_is_caught(cfg, data_dir, decomposed):
    zarr.open_group(store=str(decomposed), mode="r+")["q.modes.0"][0, 0] += 1
    assert stale_reason(decomposed, cfg, data_dir / "synthetic.zarr") == "checksum mismatch in q.modes.0"


def test_force_recomputes(cfg, data_dir, decomposed, caplog):
    with caplog.at_level(logging.INFO):
        write_decomposition(cfg, "synthetic", data_dir, decomposed.parent, force=True)
    assert "decomposing (--force)" in caplog.text
    assert verify(decomposed) == []


def test_code_hash_covers_the_vendored_spod_sources(monkeypatch):
    assert all((store.SPOD_DIR / name).is_file() for name in store.SPOD_FILES)
    full = store.code_hash()
    monkeypatch.setattr(store, "SPOD_FILES", ())
    assert store.code_hash() != full


# ---------------------------------------------------------------- atomic writes and input checks


def _crash(cfg, data_path):
    raise RuntimeError("sPOD crashed")
    yield  # a generator, like decompose_case


def test_failure_keeps_the_previous_store(cfg, data_dir, decomposed, monkeypatch):
    monkeypatch.setattr(store, "decompose_case", _crash)
    with pytest.raises(RuntimeError, match="sPOD crashed"):
        write_decomposition(cfg, "synthetic", data_dir, decomposed.parent, force=True)
    assert verify(decomposed) == []
    assert sorted(p.name for p in decomposed.parent.iterdir()) == ["synthetic.zarr"]


def test_missing_dataset_is_reported(cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="run `spodnn generate configs/synthetic.yaml` first"):
        write_decomposition(cfg, "synthetic", tmp_path / "nowhere", tmp_path / "out")


@pytest.mark.parametrize(
    ("case", "lay", "match"),
    [
        ("wildfire1d", "synthetic", "can't decompose synthetic data"),
        ("synthetic", "1d", "can't decompose wildfire data"),
        ("wildfire2d", "1d", "use_scipy_transform only works on 2D grids"),
        ("wildfire2d_ww", "2d", "shift_modes must be set for 2d_ww data and only there"),
        ("wildfire2d", "2d_ww", "shift_modes must be set for 2d_ww data and only there"),
    ],
)
def test_config_must_fit_the_dataset(case, lay, match):
    with pytest.raises(ValueError, match=match):
        check_config(load_decomposition_config(CONFIGS_DIR / "decomposition" / f"{case}.yaml"), lay)


@pytest.mark.parametrize(
    ("case", "lay"),
    [("synthetic", "synthetic"), ("wildfire1d", "1d"), ("wildfire2d", "2d"), ("wildfire2d_ww", "2d_ww")],
)
def test_every_real_config_fits_its_dataset(case, lay):
    check_config(load_decomposition_config(CONFIGS_DIR / "decomposition" / f"{case}.yaml"), lay)


# ---------------------------------------------------------------- the command


def test_decompose_command_writes_and_skips(cfg, data_dir, tmp_path, caplog):
    path = tmp_path / "synthetic.yaml"  # the store is named after the config file, as with generate
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    argv = ["decompose", str(path), "--data", str(data_dir), "--out", str(tmp_path / "out")]
    assert main(argv) == 0
    assert np.isfinite(read_manifest(tmp_path / "out" / "synthetic.zarr")["variables"]["q"]["offline_error"])
    with caplog.at_level(logging.INFO):
        main(argv)
    assert "up to date" in caplog.text


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("missing.yaml", "No such file"),
        ("wildfire1d.yaml", "no dataset at"),
    ],
)
def test_decompose_command_rejects_bad_input(tmp_path, capsys, config, message):
    path = CONFIGS_DIR / "decomposition" / config
    with pytest.raises(SystemExit) as exit_info:
        main(["decompose", str(path), "--data", str(tmp_path)])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
