#!/usr/bin/env python3
"""Build the pinned NumPy/SciPy environment that re-checks shellsim's scientific suites.

The portable suites under ``tests/python/numpy`` state literal expectations taken from the
releases pinned in ``tests/python/scientific-requirements.txt``. ``cargo test`` always runs them
under shellsim offline; this optional step confirms the expectations against real NumPy. It
creates a uv-managed virtual environment, runs the suites there with pytest, and prints the
interpreter path. Export that path as ``SHELLSIM_SCIENTIFIC_PYTHON`` to make ``cargo test`` run
the same check.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPOSITORY_ROOT / "tests/python/scientific-requirements.txt"
SUITES = (REPOSITORY_ROOT / "tests/python/numpy",)
PYTHON_VERSION = "3.14"


def run(command: Sequence[str]) -> None:
    """Run one command from the repository root, raising on failure."""

    logging.info("[scientific-reference] %s", " ".join(command))
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main(arguments: Sequence[str] | None = None) -> int:
    """Create or refresh the reference environment, then run the suites under it."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        type=Path,
        default=REPOSITORY_ROOT / "target/scientific-reference",
        help="virtual environment location (default: target/scientific-reference)",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="build the environment without running the suites",
    )
    options = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    venv = options.venv.resolve()
    python = venv / "bin/python"
    if not python.exists():
        run(["uv", "venv", "--python", PYTHON_VERSION, str(venv)])
    run(["uv", "pip", "install", "--python", str(python), "-r", str(REQUIREMENTS)])
    if not options.no_run:
        run([str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider", *map(str, SUITES)])
    print(f"export SHELLSIM_SCIENTIFIC_PYTHON={python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
