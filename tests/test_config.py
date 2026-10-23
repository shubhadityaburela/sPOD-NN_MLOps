"""Tests for spodnn.config."""

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from paper import TEST_MU, TRAIN_MU
from spodnn.config import CaseConfig, SyntheticCaseConfig, WildfireCaseConfig, config_hash, load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def validate(data):
    return TypeAdapter(CaseConfig).validate_python(data)


@pytest.fixture
def wildfire():
    return yaml.safe_load((CONFIGS / "wildfire2d_ww.yaml").read_text())


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_config_loads(path):
    cfg = load_config(path)
    expected = SyntheticCaseConfig if cfg.kind == "synthetic" else WildfireCaseConfig
    assert isinstance(cfg, expected)


def test_load_config_accepts_a_string_path():
    assert load_config(str(CONFIGS / "synthetic.yaml")).kind == "synthetic"


def test_synthetic_split_matches_the_paper():
    cfg = load_config(CONFIGS / "synthetic.yaml")
    assert cfg.mu.train == TRAIN_MU
    assert cfg.mu.test == TEST_MU


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d["physics"].update(cfI=0.8), "Extra inputs are not permitted"),
        (lambda d: d.update(stpe=10), "Extra inputs are not permitted"),
        (lambda d: d["physics"].update(cfl=1.5), "less than or equal to 1"),
        (lambda d: d["physics"].update(cfl=0), "greater than 0"),
        (lambda d: d["physics"].update(v_y=0.1), "v_y is ignored"),
        (lambda d: d["grid"].update(Nx=0), "greater than 0"),
        (lambda d: d["grid"].update(Nt=505), "must be divisible by step"),
        (lambda d: d["grid"].update(Ny=1), "edge_detect should be false"),
        (lambda d: d["grid"].update(Lx=400), "edge_detect will only work"),
        (lambda d: d["grid"].update(Ly=1000), "edge_detect will only work"),
        (lambda d: d["physics"].update(ThermalDiffusivity=-0.2), "greater than 0"),
        (lambda d: d["physics"].pop("TempRisePerSecond"), "Field required"),
        (lambda d: d["physics"].update(TAmbient=200), "greater than 273.15"),
        (lambda d: d.update(edge_detect="yes"), "valid boolean"),
        (lambda d: d.update(dtype="float16"), "'float32' or 'float64'"),
        (lambda d: d["beta"].update(train=[]), "should have values"),
        (lambda d: d["beta"].update(test=[]), "should have values"),
        (lambda d: d["beta"].update(test=[600]), "outside the training range"),
        (lambda d: d["beta"].update(train=[540, 550, 550, 580]), "strictly increasing"),
        (lambda d: d["beta"].update(train=[580, 540, 560]), "strictly increasing"),
        (lambda d: d.pop("kind"), "Unable to extract tag"),
        (lambda d: d.update(kind="wildfire3d"), "does not match any of the expected tags"),
        (lambda d: d.pop("physics"), "Field required"),
    ],
    ids=[
        "typo-in-section",
        "typo-at-top",
        "cfl-above-1",
        "cfl-zero",
        "v_y-nonzero",
        "Nx-zero",
        "Nt-not-divisible",
        "edge_detect-in-1D",
        "edge_detect-dx-not-1",
        "edge_detect-dy-not-1",
        "negative-diffusivity",
        "missing-constant",
        "ambient-below-freezing",
        "bool-as-string",
        "unknown-dtype",
        "empty-train",
        "empty-test",
        "test-outside-range",
        "duplicate-train",
        "unsorted-train",
        "missing-kind",
        "unknown-kind",
        "missing-section",
    ],
)
def test_invalid_config_is_rejected(wildfire, edit, match):
    edit(wildfire)
    with pytest.raises(ValidationError, match=match):
        validate(wildfire)


def test_config_is_read_only():
    cfg = load_config(CONFIGS / "wildfire1d.yaml")
    with pytest.raises(ValidationError, match="frozen"):
        cfg.step = 5
    with pytest.raises(ValidationError, match="frozen"):
        cfg.grid.Nt = 10


def test_hash_ignores_yaml_formatting(wildfire, tmp_path):
    # Reversed key order, flow style, comments and ints written as floats: same values, same hash.
    reordered = {key: wildfire[key] for key in reversed(list(wildfire))}
    reordered["grid"] = {**reordered["grid"], "Lx": 500.0}
    path = tmp_path / "reformatted.yaml"
    path.write_text("# a comment\n" + yaml.safe_dump(reordered, default_flow_style=True, sort_keys=False))
    assert config_hash(load_config(path)) == config_hash(load_config(CONFIGS / "wildfire2d_ww.yaml"))


def test_hash_changes_with_any_value(wildfire):
    changed = copy.deepcopy(wildfire)
    changed["beta"]["test"] = [555.0]
    assert config_hash(validate(changed)) != config_hash(validate(wildfire))


def test_hash_is_sha256_hex():
    h = config_hash(load_config(CONFIGS / "synthetic.yaml"))
    assert len(h) == 64
    assert int(h, 16) >= 0


def test_every_case_has_its_own_hash():
    hashes = {config_hash(load_config(p)) for p in CONFIGS.glob("*.yaml")}
    assert len(hashes) == len(list(CONFIGS.glob("*.yaml")))
