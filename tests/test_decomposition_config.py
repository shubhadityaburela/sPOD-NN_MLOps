"""Tests for spodnn.decomposition_config."""

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from spodnn.config import config_hash, load_config
from spodnn.decomposition_config import (
    DecompositionConfig,
    SyntheticDecompositionConfig,
    WildfireDecompositionConfig,
    load_decomposition_config,
)

CONFIGS = Path(__file__).parents[1] / "configs"
DECOMPOSITION = CONFIGS / "decomposition"


def validate(data):
    return TypeAdapter(DecompositionConfig).validate_python(data)


@pytest.fixture
def wildfire():
    return yaml.safe_load((DECOMPOSITION / "wildfire2d_ww.yaml").read_text())


@pytest.fixture
def synthetic():
    return yaml.safe_load((DECOMPOSITION / "synthetic.yaml").read_text())


@pytest.mark.parametrize("path", sorted(DECOMPOSITION.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_config_loads(path):
    cfg = load_decomposition_config(path)
    expected = SyntheticDecompositionConfig if cfg.kind == "synthetic" else WildfireDecompositionConfig
    assert isinstance(cfg, expected)


def test_load_accepts_a_string_path():
    assert load_decomposition_config(str(DECOMPOSITION / "synthetic.yaml")).kind == "synthetic"


def test_every_dataset_has_a_decomposition_config_of_its_kind():
    # configs/decomposition/<case>.yaml decomposes data/<case>.zarr, made from configs/<case>.yaml.
    data = {p.stem: load_config(p).kind for p in CONFIGS.glob("*.yaml")}
    decomposition = {p.stem: load_decomposition_config(p).kind for p in DECOMPOSITION.glob("*.yaml")}
    assert decomposition == data


def test_paper_settings():
    # The choices that differ between the cases (paper/*_sup.py, D1 and D6 of step 2.6).
    cfg = {p.stem: load_decomposition_config(p) for p in DECOMPOSITION.glob("*.yaml")}
    assert {case: c.penalty for case, c in cfg.items()} == {
        "synthetic": "matrix",
        "wildfire1d": "matrix",
        "wildfire2d": "grid",
        "wildfire2d_ww": "grid",
    }
    assert {case: c.shift_modes for case, c in cfg.items()} == {
        "synthetic": None,
        "wildfire1d": None,
        "wildfire2d": None,
        "wildfire2d_ww": 3,
    }
    assert [case for case, c in cfg.items() if c.transform.use_scipy_transform] == ["wildfire2d"]
    assert (cfg["wildfire2d"].spod.T.n_iter, cfg["wildfire2d"].spod.S.n_iter) == (15, 30)
    assert cfg["synthetic"].spod.nmodes == 8
    assert all(c.spod.T.nmodes is None and c.spod.S.nmodes is None for case, c in cfg.items() if case != "synthetic")


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d.update(penatly="grid"), "Extra inputs are not permitted"),
        (lambda d: d["spod"]["T"].update(niter=10), "Extra inputs are not permitted"),
        (lambda d: d["transform"].update(order=3), "Extra inputs are not permitted"),
        (lambda d: d["spod"].update(q={}), "Extra inputs are not permitted"),
        (lambda d: d["spod"]["T"].update(n_iter=0), "greater than 0"),
        (lambda d: d["spod"]["S"].update(mu_factor=0), "greater than 0"),
        (lambda d: d["spod"]["T"].update(lambda_factor=-5), "greater than 0"),
        (lambda d: d["spod"]["T"].update(eps=0), "greater than 0"),
        (lambda d: d["spod"]["T"].update(nmodes_max=0), "greater than 0"),
        (lambda d: d["spod"]["T"].update(nmodes=0), "greater than 0"),
        (lambda d: d["spod"]["T"].update(nmodes=11), "nmodes <= nmodes_max"),
        (lambda d: d["spod"]["T"].update(n_iter=2.5), "valid integer"),
        (lambda d: d["transform"].update(interp_order=2), "amongst"),
        (lambda d: d["transform"].update(interp_order=0), "greater than 0"),
        (lambda d: d["transform"].update(use_scipy_transform="yes"), "valid boolean"),
        (lambda d: d.update(penalty="frobenius"), "'matrix' or 'grid'"),
        (lambda d: d.update(shift_modes=0), "greater than 0"),
        (lambda d: d["spod"].pop("S"), "Field required"),
        (lambda d: d.pop("transform"), "Field required"),
        (lambda d: d.pop("penalty"), "Field required"),
        (lambda d: d.pop("kind"), "Unable to extract tag"),
        (lambda d: d.update(kind="wildfire3d"), "does not match any of the expected tags"),
    ],
    ids=[
        "typo-at-top",
        "typo-in-spod",
        "typo-in-transform",
        "unknown-variable",
        "n_iter-zero",
        "mu_factor-zero",
        "lambda_factor-negative",
        "eps-zero",
        "nmodes_max-zero",
        "nmodes-zero",
        "nmodes-above-nmodes_max",
        "n_iter-not-integer",
        "interp_order-even",
        "interp_order-zero",
        "scipy-not-boolean",
        "unknown-penalty",
        "shift_modes-zero",
        "variable-missing",
        "transform-missing",
        "penalty-missing",
        "kind-missing",
        "kind-unknown",
    ],
)
def test_bad_wildfire_config_is_rejected(wildfire, edit, match):
    edit(wildfire)
    with pytest.raises(ValidationError, match=match):
        validate(wildfire)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d.update(shift_modes=3), "Input should be None"),
        (lambda d: d.update(spod={"T": d["spod"], "S": d["spod"]}), "Extra inputs are not permitted"),
        (lambda d: d["spod"].update(nmodes=29), "nmodes <= nmodes_max"),
        (lambda d: d.pop("spod"), "Field required"),
    ],
    ids=["shift_modes-set", "per-variable-spod", "nmodes-above-nmodes_max", "spod-missing"],
)
def test_bad_synthetic_config_is_rejected(synthetic, edit, match):
    edit(synthetic)
    with pytest.raises(ValidationError, match=match):
        validate(synthetic)


def test_config_is_read_only():
    cfg = load_decomposition_config(DECOMPOSITION / "wildfire2d.yaml")
    with pytest.raises(ValidationError, match="frozen"):
        cfg.spod.T.n_iter = 30


def test_config_hash_follows_values_not_formatting(wildfire, tmp_path):
    reference = config_hash(validate(wildfire))

    # Same values written differently: block style, other key order, comments.
    path = tmp_path / "reordered.yaml"
    path.write_text("# a comment\n" + yaml.safe_dump(dict(reversed(list(wildfire.items()))), sort_keys=False))
    assert config_hash(load_decomposition_config(path)) == reference

    changed = copy.deepcopy(wildfire)
    changed["spod"]["T"]["n_iter"] += 1
    assert config_hash(validate(changed)) != reference
