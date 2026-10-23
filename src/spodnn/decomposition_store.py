"""Decompositions on disk: one zarr store per case, data/decomposition/<case>.zarr, with a manifest.

For each variable v (q for synthetic, T and S for wildfire) the store holds:

    v.modes.<k>             (N, rank_k)           sPOD modes of frame k (polar coordinates in 2D)
    v.amplitudes.<k>        (rank_k, M_train)     training amplitudes diag(sigma) V^T
    v.amplitudes_test.<k>   (rank_k, M_test)      test frames projected onto the modes
    v.shifts, v.shifts_test (n_shift_outputs, M)  the shifts the network predicts
    v.shift_basis           (N_theta, n_modes)    2D_ww only: modes of the shift POD
    v.pod.modes, v.pod.amplitudes, v.pod.amplitudes_test   the POD baseline (Cartesian)

The names are flat (no nested groups), so spodnn.dataset's verify, open_case and
read_manifest work on these stores unchanged. The manifest records the config and
its fingerprint, per variable the sPOD grid (to rebuild the transforms) and the
diagnostics, a sha256 of every array, and provenance. A store is up to date while
its config, the decomposing code (including the vendored sPOD sources), the
numerical libraries and the input dataset's content are unchanged.
"""

import hashlib
import json
import logging
import platform
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import spod
import zarr

from spodnn.config import config_hash
from spodnn.dataset import PACKAGE_DIR, _array_sha256, _git, read_manifest, swap_in, verify
from spodnn.dataset import store_path as data_store_path
from spodnn.decompose import FRAMES, check_config, decompose_case, layout

log = logging.getLogger(__name__)

SCHEMA = "spodnn.decomposition/1"

# Everything the decomposition depends on: if any of these change, the store is recomputed.
CODE_FILES = (
    "config.py",
    "decomposition_config.py",
    "decompose.py",
    "decomposition.py",
    "wildfire/polar.py",
    "wildfire/shifts.py",
)
SPOD_DIR = Path(spod.__file__).parent
SPOD_FILES = ("sPOD_tools.py", "transforms.py")  # vendored: a patch there can change the results
NUMERIC_LIBRARIES = ("numpy", "scipy", "scikit-learn", "numba", "polarTransform")


def store_path(out_dir, case):
    return Path(out_dir) / f"{case}.zarr"


def code_hash():
    """sha256 over the decomposing source files: CODE_FILES and the vendored SPOD_FILES."""
    h = hashlib.sha256()
    for root, names, prefix in ((PACKAGE_DIR, CODE_FILES, ""), (SPOD_DIR, SPOD_FILES, "spod/")):
        for name in names:
            h.update((prefix + name).encode())
            h.update((root / name).read_bytes())
    return h.hexdigest()


def data_hash(data_path):
    """sha256 over the input dataset's array checksums: it changes whenever the data's content does."""
    checksums = read_manifest(data_path)["checksums"]
    return hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()


def fingerprint(cfg, data_path):
    """What the decomposition is a function of: config, code, numerical libraries and input data."""
    return {
        "config_hash": config_hash(cfg),
        "code_hash": code_hash(),
        "libraries": {lib: version(lib) for lib in NUMERIC_LIBRARIES},
        "data_hash": data_hash(data_path),
    }


def stale_reason(path, cfg, data_path):
    """Why the store at path must be (re)computed for cfg and the data at data_path, or None if up to date."""
    manifest = read_manifest(path)
    if manifest is None:
        return "no store yet"
    if manifest.get("schema") != SCHEMA:
        return f"store schema {manifest.get('schema')} is not {SCHEMA}"
    old, new = manifest["fingerprint"], fingerprint(cfg, data_path)
    for key, reason in (
        ("config_hash", "config changed"),
        ("code_hash", "decomposing code changed"),
        ("data_hash", "input data changed"),
    ):
        if old[key] != new[key]:
            return reason
    for lib, v in new["libraries"].items():
        if old["libraries"].get(lib) != v:
            return f"{lib} {old['libraries'].get(lib)} -> {v}"
    if bad := verify(path):
        return f"checksum mismatch in {', '.join(bad)}"
    return None


def input_path(cfg, case, data_dir):
    """The dataset decomposed for `case`, after checking it exists and fits cfg (ValueError/FileNotFoundError)."""
    data_path = data_store_path(data_dir, case)
    manifest = read_manifest(data_path)
    if manifest is None:
        raise FileNotFoundError(f"no dataset at {data_path}; run `spodnn generate configs/{case}.yaml` first")
    check_config(cfg, layout(manifest["config"]))
    return data_path


def _write(group, cfg, data_path):
    """Decompose every variable into group; returns per-variable diagnostics and the array checksums."""
    variables, checksums = {}, {}

    def put(name, values):
        group.create_array(name, data=np.asarray(values, dtype=np.float64))
        checksums[name] = _array_sha256(group[name])

    start = time.perf_counter()
    for var, result in decompose_case(cfg, data_path):
        runtime = time.perf_counter() - start
        v, s = var.name, result.spod
        for k, (U, A, A_test) in enumerate(zip(s.modes, s.time_amplitudes, s.time_amplitudes_test, strict=True)):
            put(f"{v}.modes.{k}", U)
            put(f"{v}.amplitudes.{k}", A)
            put(f"{v}.amplitudes_test.{k}", A_test)
        put(f"{v}.shifts", result.shifts_train)
        put(f"{v}.shifts_test", result.shifts_test)
        if result.shift_basis is not None:
            put(f"{v}.shift_basis", result.shift_basis)
        put(f"{v}.pod.modes", result.pod.modes)
        put(f"{v}.pod.amplitudes", result.pod.time_amplitudes)
        put(f"{v}.pod.amplitudes_test", result.pod.time_amplitudes_test)

        test = result.test_spod
        variables[v] = {
            "layout": var.layout,
            "frames": FRAMES[var.layout],
            "grid_shape": list(var.grid_shape),
            "domain": [float(d) for d in var.domain],
            "spacing": [float(d) for d in var.spacing],
            "ranks": s.ranks,
            "pod_modes": int(result.pod.modes.shape[1]),
            "interpolation_error": float(s.interpolation_error),
            "offline_error": float(s.offline_error),
            "error_history": [float(e) for e in s.error_history],
            "test_spod": None
            if test is None
            else {
                "ranks": test.ranks,
                "interpolation_error": float(test.interpolation_error),
                "offline_error": float(test.offline_error),
            },
            "runtime_s": round(runtime, 3),
        }
        log.info("  %s: ranks %s, offline error %.3e, %.1f s", v, s.ranks, s.offline_error, runtime)
        start = time.perf_counter()
    return variables, checksums


def write_decomposition(cfg, case, data_dir, out_dir, force=False):
    """Decompose data_dir/<case>.zarr into out_dir/<case>.zarr unless it is up to date (or force); returns the path."""
    data_path = input_path(cfg, case, data_dir)
    path = store_path(out_dir, case)
    reason = "--force" if force else stale_reason(path, cfg, data_path)
    if reason is None:
        log.info("%s: up to date, skipping (%s)", case, path)
        return path
    log.info("%s: decomposing (%s)", case, reason)

    git = _git()
    if git["dirty"]:
        log.warning("%s: the working tree has uncommitted changes; the manifest records git_dirty = true", case)

    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    try:
        group = zarr.open_group(store=str(tmp), mode="w")
        fp = fingerprint(cfg, data_path)
        variables, checksums = _write(group, cfg, data_path)
        group.attrs["manifest"] = {
            "schema": SCHEMA,
            "case": case,
            "kind": cfg.kind,
            "config": cfg.model_dump(mode="json"),
            "fingerprint": fp,
            "data": str(data_path),
            "variables": variables,
            "checksums": checksums,
            "provenance": {
                "created_utc": started.isoformat(timespec="seconds"),
                "git": git,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "versions": {lib: version(lib) for lib in ("spodnn", "zarr", *NUMERIC_LIBRARIES)},
            },
        }
    except BaseException:  # also Ctrl-C: never leave a half-written store behind
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    swap_in(tmp, path)
    log.info("%s: wrote %s in %.1f s", case, path, sum(v["runtime_s"] for v in variables.values()))
    return path
