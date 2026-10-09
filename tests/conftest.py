"""Shared pytest fixtures. pytest loads this file automatically"""

import json
from pathlib import Path

import pytest

import paper
from spodnn.decomposition import pod, spod
from spodnn.synthetic import Grid, generate

GOLDEN_DIR = Path(__file__).parent / "golden"


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
