"""The `spodnn` command (installed through [project.scripts] in pyproject.toml).

spodnn generate configs/wildfire1d.yaml [--out data] [--force]
spodnn decompose configs/decomposition/wildfire1d.yaml [--data data] [--out data/decomposition] [--force]
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from spodnn.config import load_config
from spodnn.dataset import write_case
from spodnn.decomposition_config import load_decomposition_config
from spodnn.decomposition_store import input_path, write_decomposition


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="spodnn",
        description="sPOD-NN command line interface.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        description="Valid SPODNN operations",
        required=True,
    )

    # 'generate' subcommand parser
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic or wildfire data based on configuration.",
    )
    generate_parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML configuration file.",
    )
    generate_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Output directory path (defaults to 'data').",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the stored data is up to date.",
    )

    # 'decompose' subcommand parser
    decompose_parser = subparsers.add_parser(
        "decompose",
        help="Decompose a generated dataset (sPOD and POD) based on a decomposition configuration.",
    )
    decompose_parser.add_argument(
        "config",
        type=Path,
        help="Path to the decomposition YAML file; configs/decomposition/<case>.yaml decomposes <data>/<case>.zarr.",
    )
    decompose_parser.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help="Directory of the generated datasets (defaults to 'data').",
    )
    decompose_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/decomposition"),
        help="Output directory path (defaults to 'data/decomposition').",
    )
    decompose_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the stored decomposition is up to date.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main CLI entry point.

    Parameters
    ----------
    argv : Sequence[str] | None
        List of command-line arguments. If None, argparse defaults to sys.argv[1:].
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "generate":
        try:
            cfg = load_config(args.config)
        except (OSError, yaml.YAMLError, ValidationError) as e:
            parser.error(f"cannot load {args.config}: {e}")  # exits with status 2
        write_case(cfg, case=args.config.stem, out_dir=args.out, force=args.force)

    if args.command == "decompose":
        case = args.config.stem
        try:
            cfg = load_decomposition_config(args.config)
            input_path(cfg, case, args.data)  # the dataset exists and fits the config
        except (OSError, yaml.YAMLError, ValidationError, ValueError) as e:
            parser.error(f"cannot decompose {args.config}: {e}")  # exits with status 2
        write_decomposition(cfg, case=case, data_dir=args.data, out_dir=args.out, force=args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
