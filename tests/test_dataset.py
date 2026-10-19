"""Tests for spodnn.dataset: the zarr store, its manifest, skipping and atomic writes."""

import logging

import numpy as np
import pytest
import yaml
import zarr
from conftest import CONFIGS_DIR, small_wildfire
from numpy.testing import assert_array_equal
from pydantic import TypeAdapter

import spodnn.dataset as dataset
from spodnn.config import CaseConfig
from spodnn.dataset import CODE_FILES, PACKAGE_DIR, open_case, read_manifest, stale_reason, verify, write_case
from spodnn.generate import generate_case


def small_synthetic():
    data = yaml.safe_load((CONFIGS_DIR / "synthetic.yaml").read_text())
    data["grid"].update(Nx=64, Nt=32)
    return TypeAdapter(CaseConfig).validate_python(data)


@pytest.fixture(scope="module")
def cfg_1d():
    return small_wildfire("wildfire1d", Lx=200, Nx=400, Nt=400)


@pytest.fixture
def store_1d(cfg_1d, tmp_path):
    return write_case(cfg_1d, "fire1d", tmp_path)


def test_store_holds_every_sample_in_config_order(cfg_1d, store_1d):
    group = zarr.open_group(store=str(store_1d), mode="r")
    samples = list(generate_case(cfg_1d))
    assert_array_equal(group["params"][:], [*cfg_1d.beta.train, *cfg_1d.beta.test])
    assert group["snapshots"].dtype == np.float32
    assert group["shifts"].dtype == np.float64
    for i, sample in enumerate(samples):
        assert_array_equal(group["snapshots"][i], sample.q.astype(np.float32))
        assert_array_equal(group["shifts"][i], sample.shifts)
    for name in ("x", "y", "t"):
        assert_array_equal(group[name][:], getattr(samples[0], name))
    assert "frames" not in group


def test_manifest_records_provenance(cfg_1d, store_1d):
    manifest = read_manifest(store_1d)
    assert manifest["schema"] == dataset.SCHEMA
    assert manifest["case"] == "fire1d"
    assert manifest["config"] == cfg_1d.model_dump(mode="json")
    assert manifest["fingerprint"] == dataset.fingerprint(cfg_1d)
    assert manifest["test_indices"] == [5]  # five training betas, then 558.49
    assert len(manifest["provenance"]["runtime_s"]) == 6
    assert set(manifest["checksums"]) == {"snapshots", "shifts", "x", "y", "t", "params"}
    assert {"commit", "dirty"} <= set(manifest["provenance"]["git"])


def test_synthetic_store_keeps_the_frames(tmp_path):
    cfg = small_synthetic()
    group = open_case(write_case(cfg, "synthetic", tmp_path))
    n = len(cfg.mu.train) + len(cfg.mu.test)
    assert group["snapshots"].shape == (n, 64, 32)
    assert group["frames"].shape == (n, 2, 64, 32)
    assert group["shifts"].shape == (n, 2, 32)


def test_2d_ww_store_keeps_the_compact_shifts(tmp_path):
    cfg = small_wildfire("wildfire2d_ww", Lx=100, Ly=100, Nx=100, Ny=100, Nt=150)
    group = open_case(write_case(cfg, "ww", tmp_path))
    assert group["shifts"].shape == (6, 100, 15)  # (n, N_theta, Nt), not (n, 2, 2, N_r, N_theta, Nt)


def test_large_snapshots_are_chunked_per_parameter_value():
    assert dataset._snapshot_chunks(500_000, 90) == (1, dataset.CHUNK_BYTES // (90 * 4), 90)
    assert dataset._snapshot_chunks(800, 40) == (1, 800, 40)


# Checksums


def test_fresh_store_verifies(store_1d):
    assert verify(store_1d) == []


def test_corruption_is_caught(cfg_1d, store_1d):
    zarr.open_group(store=str(store_1d), mode="r+")["snapshots"][2, 10, 3] += 1
    assert verify(store_1d) == ["snapshots"]
    assert stale_reason(store_1d, cfg_1d) == "checksum mismatch in snapshots"
    with pytest.raises(ValueError, match="checksum mismatch"):
        open_case(store_1d)


# Skipping unchanged work


def _fail_if_called(cfg):
    raise AssertionError("generate_case should not run")


def test_up_to_date_store_is_skipped(cfg_1d, store_1d, monkeypatch, caplog):
    monkeypatch.setattr(dataset, "generate_case", _fail_if_called)
    with caplog.at_level(logging.INFO):
        assert write_case(cfg_1d, "fire1d", store_1d.parent) == store_1d
    assert "up to date" in caplog.text


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda mp, cfg: (cfg.model_copy(update={"step": 20}), cfg)[0], "config changed"),
        (lambda mp, cfg: (mp.setattr(dataset, "code_hash", lambda: "0" * 64), cfg)[1], "generating code changed"),
        (lambda mp, cfg: (mp.setattr(dataset, "version", lambda lib: "0.0"), cfg)[1], "numpy"),
    ],
    ids=["config", "code", "library"],
)
def test_changes_trigger_regeneration(cfg_1d, store_1d, monkeypatch, change, reason):
    cfg = change(monkeypatch, cfg_1d)
    assert reason in stale_reason(store_1d, cfg)


def test_force_regenerates(cfg_1d, store_1d, caplog):
    with caplog.at_level(logging.INFO):
        write_case(cfg_1d, "fire1d", store_1d.parent, force=True)
    assert "generating (--force)" in caplog.text
    assert verify(store_1d) == []


def test_missing_or_foreign_store_is_regenerated(cfg_1d, tmp_path):
    assert stale_reason(tmp_path / "none.zarr", cfg_1d) == "no store yet"
    zarr.open_group(store=str(tmp_path / "other.zarr"), mode="w")  # a zarr store without our manifest
    assert stale_reason(tmp_path / "other.zarr", cfg_1d) == "no store yet"


def test_code_hash_covers_every_generating_module():
    # A new module under wildfire/ must be added to CODE_FILES, or its changes won't trigger regeneration.
    wildfire = {f"wildfire/{p.name}" for p in (PACKAGE_DIR / "wildfire").glob("*.py") if p.name != "__init__.py"}
    assert wildfire <= set(CODE_FILES)
    assert all((PACKAGE_DIR / name).is_file() for name in CODE_FILES)


# Atomic writes


def _broken_after_one(cfg):
    samples = generate_case(cfg)
    yield next(samples)
    raise RuntimeError("solver crashed")


def test_failure_leaves_no_store(cfg_1d, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "generate_case", _broken_after_one)
    with pytest.raises(RuntimeError, match="solver crashed"):
        write_case(cfg_1d, "fire1d", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_failure_keeps_the_previous_store(cfg_1d, store_1d, monkeypatch):
    monkeypatch.setattr(dataset, "generate_case", _broken_after_one)
    with pytest.raises(RuntimeError):
        write_case(cfg_1d, "fire1d", store_1d.parent, force=True)
    assert verify(store_1d) == []
    assert sorted(p.name for p in store_1d.parent.iterdir()) == ["fire1d.zarr"]
