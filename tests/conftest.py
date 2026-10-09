"""Shared pytest fixtures. pytest loads this file automatically"""

import json
from pathlib import Path
import pytest
from spodnn.synthetic import generate, Grid

GOLDEN_DIR = Path(__file__).parent / "golden"

TRAIN_MU = [0.1, 0.15, 0.2, 0.25, 0.3]
TEST_MU = [0.23]

grid = Grid()  # create a grid instance for the tests


@pytest.fixture(scope="session")
def golden():
    # Reference results from the Paper/ folder
    return json.loads((GOLDEN_DIR / "synthetic.json").read_text())


@pytest.fixture(scope="session")
def train_set():
    return generate(TRAIN_MU, grid, n_terms=8)        # built once for all tests


@pytest.fixture(scope="session")
def test_set():
    return generate(TEST_MU, grid, n_terms=8)     # built once for all tests
