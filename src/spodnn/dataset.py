"""Datasets on disk: one zarr store per case, with a provenance manifest.

data/<case>.zarr holds, for the n parameter values in config order (train,
then test):

    params     (n,)                   the β or μ values
    snapshots  (n, rows, Nt) float32  chunked per parameter value and block of rows
    shifts     (n, ...)      float64
    frames     (n, 2, Nx, Nt) float32 synthetic only
    x, y, t                           grid and time labels

The root attributes hold the manifest: config and its hash, a hash of the
generating code, library versions, git state, runtimes and a sha256 of every
array's content. write_case skips a case whose store is up to date and writes
new stores atomically: into <case>.zarr.tmp first, renamed when complete.
"""

import hashlib
import logging
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import zarr

from spodnn.config import config_hash
from spodnn.generate import generate_case

log = logging.getLogger(__name__)

SCHEMA = "spodnn.dataset/1"
PACKAGE_DIR = Path(__file__).parent
REPO_DIR = PACKAGE_DIR.parents[1]

# Everything the data depends on: if any of these change, the store is regenerated.
CODE_FILES = (
    "config.py",
    "generate.py",
    "synthetic.py",
    "wildfire/operators.py",
    "wildfire/polar.py",
    "wildfire/shifts.py",
    "wildfire/solver.py",
)
NUMERIC_LIBRARIES = ("numpy", "scipy", "scikit-image", "polarTransform")

CHUNK_BYTES = 32 * 2**20  # ~32 MB per snapshot chunk, uncompressed
SNAPSHOT_DTYPE = np.float32


def code_hash():
    """sha256 over the source files in CODE_FILES (path and content)."""
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode())
        h.update((PACKAGE_DIR / name).read_bytes())
    return h.hexdigest()


def fingerprint(cfg):
    """What the data is a function of: the config, the code and the numerical libraries."""
    return {
        "config_hash": config_hash(cfg),
        "code_hash": code_hash(),
        "libraries": {lib: version(lib) for lib in NUMERIC_LIBRARIES},
    }


def _git():
    def run(*args):
        return subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True, check=True).stdout

    try:
        return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(run("status", "--porcelain").strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _array_sha256(array):
    """sha256 of an array's content in C order, read one slice of the first axis at a time."""
    h = hashlib.sha256()
    if array.ndim == 0 or array.shape[0] == 0:
        h.update(np.ascontiguousarray(array[...]).tobytes())
    else:
        for i in range(array.shape[0]):
            h.update(np.ascontiguousarray(array[i]).tobytes())
    return h.hexdigest()


def store_path(out_dir, case):
    return Path(out_dir) / f"{case}.zarr"


def read_manifest(path):
    """The manifest of a store, or None if there is no readable store."""
    try:
        return dict(zarr.open_group(store=str(path), mode="r").attrs)["manifest"]
    except (FileNotFoundError, KeyError, ValueError, zarr.errors.GroupNotFoundError):
        return None


def verify(path):
    """Recompute every array's sha256 and compare with the manifest; returns the mismatching names."""
    group = zarr.open_group(store=str(path), mode="r")
    expected = dict(group.attrs)["manifest"]["checksums"]
    actual = {name: _array_sha256(group[name]) for name in group.array_keys()}
    return sorted(name for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name))


def stale_reason(path, cfg):
    """Why the store at path must be (re)generated for cfg, or None if it is up to date."""
    manifest = read_manifest(path)
    if manifest is None:
        return "no store yet"
    if manifest.get("schema") != SCHEMA:
        return f"store schema {manifest.get('schema')} is not {SCHEMA}"
    old, new = manifest["fingerprint"], fingerprint(cfg)
    if old["config_hash"] != new["config_hash"]:
        return "config changed"
    if old["code_hash"] != new["code_hash"]:
        return "generating code changed"
    for lib, v in new["libraries"].items():
        if old["libraries"].get(lib) != v:
            return f"{lib} {old['libraries'].get(lib)} -> {v}"
    if bad := verify(path):
        return f"checksum mismatch in {', '.join(bad)}"
    return None


def open_case(path, check=True):
    """Open a store read-only; with check=True, refuse it if any checksum doesn't match."""
    if check and (bad := verify(path)):
        raise ValueError(f"{path}: checksum mismatch in {', '.join(bad)}")
    return zarr.open_group(store=str(path), mode="r")


def _snapshot_chunks(rows, nt):
    return (1, max(1, min(rows, CHUNK_BYTES // (nt * np.dtype(SNAPSHOT_DTYPE).itemsize))), nt)


def _write(group, cfg, params):
    """Generate every sample of cfg into group; returns per-parameter runtimes and the array checksums."""
    hashes = {name: hashlib.sha256() for name in ("snapshots", "shifts", "frames")}
    runtimes = []
    samples = generate_case(cfg)
    for i, param in enumerate(params):
        start = time.perf_counter()
        sample = next(samples)
        runtimes.append(round(time.perf_counter() - start, 3))
        if i == 0:
            rows, nt = sample.q.shape
            group.create_array(
                "snapshots", shape=(len(params), rows, nt), chunks=_snapshot_chunks(rows, nt), dtype=SNAPSHOT_DTYPE
            )
            group.create_array(
                "shifts", shape=(len(params), *sample.shifts.shape), chunks=(1, *sample.shifts.shape), dtype=np.float64
            )
            if sample.frames is not None:
                frames_shape = sample.frames.shape
                group.create_array(
                    "frames", shape=(len(params), *frames_shape), chunks=(1, *frames_shape), dtype=SNAPSHOT_DTYPE
                )
            for name in ("x", "y", "t"):
                group.create_array(name, data=np.asarray(getattr(sample, name), dtype=np.float64))
        stored = {"snapshots": sample.q.astype(SNAPSHOT_DTYPE), "shifts": sample.shifts.astype(np.float64)}
        if sample.frames is not None:
            stored["frames"] = sample.frames.astype(SNAPSHOT_DTYPE)
        for name, values in stored.items():
            group[name][i] = values
            hashes[name].update(np.ascontiguousarray(values).tobytes())
        log.info(
            "  %s = %g: %d/%d in %.1f s",
            "beta" if cfg.kind == "wildfire" else "mu",
            param,
            i + 1,
            len(params),
            runtimes[-1],
        )

    group.create_array("params", data=np.asarray(params, dtype=np.float64))
    checksums = {name: h.hexdigest() for name, h in hashes.items() if name in group}
    checksums |= {name: _array_sha256(group[name]) for name in ("x", "y", "t", "params")}
    return runtimes, checksums


def write_case(cfg, case, out_dir, force=False):
    """Generate the dataset for cfg into out_dir/<case>.zarr unless it is up to date (or force); returns the path."""
    path = store_path(out_dir, case)
    reason = "--force" if force else stale_reason(path, cfg)
    if reason is None:
        log.info("%s: up to date, skipping (%s)", case, path)
        return path
    log.info("%s: generating (%s)", case, reason)

    split = cfg.beta if cfg.kind == "wildfire" else cfg.mu
    params = [*split.train, *split.test]
    git = _git()
    if git["dirty"]:
        log.warning("%s: the working tree has uncommitted changes; the manifest records git_dirty = true", case)

    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    try:
        group = zarr.open_group(store=str(tmp), mode="w")
        runtimes, checksums = _write(group, cfg, params)
        group.attrs["manifest"] = {
            "schema": SCHEMA,
            "case": case,
            "kind": cfg.kind,
            "config": cfg.model_dump(mode="json"),
            "fingerprint": fingerprint(cfg),
            "params": params,
            "test_indices": list(range(len(split.train), len(params))),
            "checksums": checksums,
            "provenance": {
                "created_utc": started.isoformat(timespec="seconds"),
                "runtime_s": runtimes,
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
    log.info("%s: wrote %s in %.1f s", case, path, sum(runtimes))
    return path


def swap_in(tmp, path):
    """Replace the store at path by the complete one at tmp; the old store is only deleted once the new one is in place."""
    old = path.with_name(path.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if path.exists():
        path.rename(old)
    tmp.rename(path)
    shutil.rmtree(old, ignore_errors=True)
