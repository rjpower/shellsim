#!/usr/bin/env python3
"""Run shellsim's repository-standard formatting and static-analysis gates.

The command is intentionally dependency-free and is shared by local development, the optional
Git hook, and CI. Rust's formatter and semantic checks operate on the whole crate, so
``--changed-files`` is an ergonomic alias rather than a reduced validation mode.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    """One deterministic subprocess check and its optional environment additions."""

    name: str
    command: tuple[str, ...]
    environment: Mapping[str, str] | None = None


def run_check(check: Check) -> bool:
    """Run one check from the repository root and return whether it succeeded."""

    logging.info("[%s] %s", check.name, " ".join(check.command))
    environment = os.environ.copy()
    if check.environment:
        environment.update(check.environment)
    completed = subprocess.run(
        check.command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        logging.error("[%s] failed with status %d", check.name, completed.returncode)
        return False
    return True


def checks(*, fix: bool) -> Sequence[Check]:
    """Build the ordered lint plan, optionally applying rustfmt before verifying it."""

    format_command = ("cargo", "fmt", "--all")
    adapter_format_command = (
        "cargo",
        "fmt",
        "--manifest-path",
        "python/native/Cargo.toml",
    )
    if not fix:
        format_command += ("--", "--check")
        adapter_format_command += ("--", "--check")
    return (
        Check("rustfmt", format_command),
        Check("python adapter rustfmt", adapter_format_command),
        Check(
            "clippy",
            (
                "cargo",
                "clippy",
                "--locked",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ),
        ),
        Check(
            "rustdoc",
            ("cargo", "doc", "--locked", "--no-deps", "--all-features"),
            {"RUSTDOCFLAGS": "-D warnings"},
        ),
        Check(
            "python adapter clippy",
            (
                "cargo",
                "clippy",
                "--locked",
                "--manifest-path",
                "python/native/Cargo.toml",
                "--",
                "-D",
                "warnings",
            ),
        ),
        Check(
            "python syntax",
            (
                sys.executable,
                "-m",
                "py_compile",
                "python/shellsim/__init__.py",
                "python/shellsim/_api.py",
                "python/shellsim/_cli.py",
                "python/shellsim/__main__.py",
                "python_tests/test_api.py",
                "python_tests/test_cli.py",
            ),
        ),
        Check("unstaged whitespace", ("git", "diff", "--check")),
        Check("staged whitespace", ("git", "diff", "--cached", "--check")),
    )


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    """Parse the Marin-compatible scope flags and shellsim's formatting option."""

    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all-files", action="store_true", help="validate the complete crate")
    scope.add_argument(
        "--changed-files",
        action="store_true",
        help="accepted for hook compatibility; Rust checks still validate the complete crate",
    )
    parser.add_argument("--fix", action="store_true", help="apply rustfmt before other checks")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run every lint check and return a non-zero status if any check fails."""

    options = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    succeeded = True
    for check in checks(fix=options.fix):
        succeeded = run_check(check) and succeeded
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
