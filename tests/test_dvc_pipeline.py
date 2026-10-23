"""Consistency of dvc.yaml with the code: DVC and spodnn must agree on what the data and its decomposition depend on.

DVC reruns a stage when one of its deps changes; `spodnn generate` skips a case
unless dataset.CODE_FILES (or the config, or the libraries) changed. If the two
lists drift apart, one of them misses a change and the data silently goes stale.
"""

from pathlib import Path

import pytest
import yaml

import spodnn.decomposition_store as decomposition_store
from spodnn.cli import build_parser
from spodnn.dataset import CODE_FILES, store_path
from spodnn.dataset import store_path as dataset_store_path

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


# The decompose stage: the same agreement between DVC and `spodnn decompose`.


@pytest.fixture(scope="module")
def decompose_stage():
    return yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]["decompose"]


def test_decompose_code_deps_match_the_store_code_files(decompose_stage):
    deps = decompose_stage["do"]["deps"]
    assert {d.removeprefix("src/spodnn/") for d in deps if d.startswith("src/")} == set(decomposition_store.CODE_FILES)
    spod_deps = {d for d in deps if d.startswith("sPOD/")}
    assert spod_deps == {
        str((decomposition_store.SPOD_DIR / f).relative_to(REPO)) for f in decomposition_store.SPOD_FILES
    }


def test_decompose_depends_on_its_dataset_config_and_lockfile(decompose_stage, generate_stage):
    deps = decompose_stage["do"]["deps"]
    assert str(dataset_store_path("data", "${item}")) in deps  # generate's output: links the two stages
    assert generate_stage["do"]["outs"][0] in deps
    assert "configs/decomposition/${item}.yaml" in deps
    assert "uv.lock" in deps


def test_every_decompose_dep_exists(decompose_stage):
    # The datasets are made by generate (and absent in CI); everything else is in the repository.
    for item in decompose_stage["foreach"]:
        for dep in decompose_stage["do"]["deps"]:
            if not dep.startswith("data/"):
                assert (REPO / dep.replace("${item}", item)).is_file(), dep


def test_every_dataset_is_decomposed(decompose_stage, generate_stage):
    configs = sorted(p.stem for p in (REPO / "configs" / "decomposition").glob("*.yaml"))
    assert sorted(decompose_stage["foreach"]) == configs == sorted(generate_stage["foreach"])


def test_decompose_command_and_output_follow_spodnn_naming(decompose_stage):
    stage = decompose_stage["do"]
    assert stage["cmd"] == "spodnn decompose configs/decomposition/${item}.yaml"
    defaults = build_parser().parse_args(["decompose", "x.yaml"])  # the command relies on its defaults
    assert (defaults.data, defaults.out) == (Path("data"), Path("data/decomposition"))
    assert stage["outs"] == [str(decomposition_store.store_path(defaults.out, "${item}"))]
