#!/usr/bin/env python3
"""Run every safe shellsim test through the command used by CI.

Shellsim's suite is local and deterministic, so the safe selection is the complete set of Cargo
targets and features. Reference-binary differential tests retain their own availability checks.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the safe test suite, forwarding optional arguments after Cargo's ``--`` separator."""

    forwarded = list(sys.argv[1:] if arguments is None else arguments)
    command = ["cargo", "test", "--locked", "--all-targets", "--all-features"]
    if forwarded:
        command.extend(["--", *forwarded])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("[tests] %s", " ".join(command))
    environment = os.environ.copy()
    environment.setdefault("RUST_BACKTRACE", "1")
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
