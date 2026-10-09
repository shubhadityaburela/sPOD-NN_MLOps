# sPOD-NN

Non-intrusive model order reduction of transport-dominated problems: a shifted POD
(sPOD) splits the snapshots into co-moving frames, and a neural network maps the
parameters `(t, mu)` to the frames' time amplitudes and shifts. This repository
turns the code of the paper into a tested Python package.

## Layout

| Folder | Content |
|---|---|
| `src/spodnn/` | The package: synthetic data, POD/sPOD decomposition, scaling, network (Lightning), sPOD-I interpolation, error metrics |
| `tests/` | pytest suite; `tests/golden/` holds reference values from the paper code |
| `scripts/make_golden.py` | Regenerates the reference values from `paper/` (needs the original Python 3.9 environment) |
| `paper/` | The paper's notebooks, helpers and trained weights; frozen reference, not modified |
| `sPOD/` | Vendored sPOD library, installed as `spod-vendored` (import `spod`), with small documented fixes |
| `DL-ROM/`, `WildlandFire_model/` | Original network code and wildland fire solver, kept for reference |

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and all dependencies come from `uv.lock`.

```bash
uv sync                          # create .venv from the lockfile
uv run pre-commit install        # run lint and format checks on every commit

uv run pytest -m "not slow"      # fast tests (seconds)
uv run pytest                    # all tests, including the sPOD runs (~1 min)

uv run ruff check                # lint
uv run ruff format               # format
```

Dependencies change only through `uv add` / `uv lock` and a committed `uv.lock`.

CI (GitHub Actions) runs the pre-commit hooks and the full test suite on every push
and pull request.
