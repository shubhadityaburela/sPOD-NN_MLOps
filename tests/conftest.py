"""Shared pytest fixtures. pytest loads this file automatically"""

import json
from pathlib import Path
import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"

@pytest.fixture(scope="session")
def golden():
    # Reference results from the Paper/ folder
    return json.loads((GOLDEN_DIR / "synthetic.json").read_text())