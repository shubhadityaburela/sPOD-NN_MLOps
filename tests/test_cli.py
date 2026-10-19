"""Tests for the spodnn command line."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import small_wildfire

from spodnn.cli import main
from spodnn.dataset import read_manifest


@pytest.fixture
def small_config(tmp_path):
    path = tmp_path / "fire1d.yaml"
    cfg = small_wildfire("wildfire1d", Lx=200, Nx=400, Nt=400)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return path


def test_generate_writes_a_store_named_after_the_config(small_config, tmp_path):
    assert main(["generate", str(small_config), "--out", str(tmp_path / "data")]) == 0
    assert read_manifest(tmp_path / "data" / "fire1d.zarr")["case"] == "fire1d"


def test_generate_skips_then_forces(small_config, tmp_path, caplog):
    out = str(tmp_path / "data")
    main(["generate", str(small_config), "--out", out])
    with caplog.at_level(logging.INFO):
        main(["generate", str(small_config), "--out", out])
        assert "up to date" in caplog.text
        main(["generate", str(small_config), "--out", out, "--force"])
        assert "generating (--force)" in caplog.text


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "required: command"),
        (["generate", "missing.yaml"], "cannot load missing.yaml"),
        (["frobnicate"], "invalid choice"),
    ],
    ids=["no-command", "missing-config", "unknown-command"],
)
def test_bad_usage_exits_with_status_2(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_invalid_config_exits_with_status_2(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: wildfire\nstpe: 10\n")
    with pytest.raises(SystemExit) as exit_info:
        main(["generate", str(bad)])
    assert exit_info.value.code == 2
    assert "cannot load" in capsys.readouterr().err


def test_entry_point_is_installed():
    # [project.scripts] put an executable next to this Python.
    script = Path(sys.executable).parent / "spodnn"
    result = subprocess.run([str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "generate" in result.stdout
