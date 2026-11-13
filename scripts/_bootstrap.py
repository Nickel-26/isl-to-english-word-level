"""Shared bootstrap for scripts: make ``src/`` importable without installing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_CONFIG = ROOT / "configs" / "include50.yaml"


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: inferred from the config path)",
    )
