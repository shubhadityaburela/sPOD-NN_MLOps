"""Consistency of dvc.yaml with the code: DVC and spodnn must agree on what the data depends on.

DVC reruns a stage when one of its deps changes; `spodnn generate` skips a case
unless dataset.CODE_FILES (or the config, or the libraries) changed. If the two
lists drift apart, one of them misses a change and the data silently goes stale.
"""

from pathlib import Path

import pytest
import yaml

from spodnn.dataset import CODE_FILES, store_path

REPO = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def generate_stage():
    return yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]["generate"]


def test_code_deps_match_dataset_code_files(generate_stage):
    deps = generate_stage["do"]["deps"]
    code_deps = {d.removeprefix("src/spodnn/") for d in deps if d.startswith("src/")}
    assert code_deps == set(CODE_FILES)


def test_config_and_lockfile_are_deps(generate_stage):
    deps = generate_stage["do"]["deps"]
    assert "configs/${item}.yaml" in deps
    assert "uv.lock" in deps  # a library upgrade must regenerate the data


def test_every_dep_exists(generate_stage):
    for item in generate_stage["foreach"]:
        for dep in generate_stage["do"]["deps"]:
            assert (REPO / dep.replace("${item}", item)).is_file(), dep


def test_every_config_has_a_stage(generate_stage):
    # A new configs/<case>.yaml without a stage would never be versioned.
    assert sorted(generate_stage["foreach"]) == sorted(p.stem for p in (REPO / "configs").glob("*.yaml"))


def test_command_and_output_follow_spodnn_naming(generate_stage):
    # spodnn generate names the store after the config file; DVC must expect that same path.
    stage = generate_stage["do"]
    assert stage["cmd"] == "spodnn generate configs/${item}.yaml"
    assert stage["outs"] == [str(store_path("data", "${item}"))]
