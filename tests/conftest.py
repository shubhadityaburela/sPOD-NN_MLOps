"""Shared pytest fixtures. pytest loads this file automatically"""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

import paper
from spodnn.config import CaseConfig
from spodnn.decomposition import pod, spod
from spodnn.synthetic import Grid, generate
from spodnn.wildfire.solver import solve

GOLDEN_DIR = Path(__file__).parent / "golden"
CONFIGS_DIR = Path(__file__).parents[1] / "configs"
TEST_BETA = 558.49


grid = Grid()  # create a grid instance for the tests


@pytest.fixture(scope="session")
def golden():
    # Reference results from the Paper/ folder
    return json.loads((GOLDEN_DIR / "synthetic.json").read_text())


@pytest.fixture(scope="session")
def train_set():
    return generate(paper.TRAIN_MU, grid, n_terms=8)  # built once for all tests


@pytest.fixture(scope="session")
def test_set():
    return generate(paper.TEST_MU, grid, n_terms=8)  # built once for all tests


@pytest.fixture(scope="session")
def pod_result(train_set, test_set):
    return pod(train_set.q, test_set.q, n_modes=paper.POD_MODES)


@pytest.fixture(scope="session")
def spod_result(train_set, test_set):
    """sPOD at the paper's configuration (~1 minute); shared by every test that needs it."""
    return spod(train_set.q, test_set.frames, train_set.transforms, n_modes=paper.SPOD_MODES, n_iter=paper.SPOD_ITER)


def small_wildfire(name, **grid):
    """A real case config with a smaller grid, validated again (so dx = dy = 1 still holds for edge_detect)."""
    data = yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text())
    data["grid"].update(grid)
    return TypeAdapter(CaseConfig).validate_python(data)


def _run(cfg):
    return cfg, solve(cfg.grid, cfg.physics, TEST_BETA, cfg.step)


# Small enough for a fraction of a second each, large enough for the fronts to travel ~25 cells.
@pytest.fixture(scope="session")
def fire_1d():
    return _run(small_wildfire("wildfire1d", Lx=200, Nx=400, Nt=400))


@pytest.fixture(scope="session")
def fire_2d():
    return _run(small_wildfire("wildfire2d", Lx=100, Ly=100, Nx=100, Ny=100, Nt=150))


@pytest.fixture(scope="session")
def fire_2d_ww():
    return _run(small_wildfire("wildfire2d_ww", Lx=100, Ly=100, Nx=100, Ny=100, Nt=150))
