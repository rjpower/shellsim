#!/usr/bin/env python3
"""Run shellsim's repository-standard formatting and static-analysis gates.

The command is shared by local development, the optional Git hook, and CI. Rust's formatter and
semantic checks operate on the whole crate. Python checks use the exact marin-style revision
pinned by ``infra/marin-style.py`` and honor the requested file scope.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    """One subprocess operation with optional environment and post-fix verification."""

    name: str
    command: tuple[str, ...]
    environment: Mapping[str, str] | None = None
    verification_command: tuple[str, ...] | None = None


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
    if completed.returncode != 0 and check.verification_command:
        logging.info(
            "[%s] fix command returned non-zero; verifying the resulting tree",
            check.name,
        )
        completed = subprocess.run(
            check.verification_command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        )
    if completed.returncode != 0:
        logging.error("[%s] failed with status %d", check.name, completed.returncode)
        return False
    return True


def marin_style_command(*, all_files: bool, fix: bool = False) -> tuple[str, ...]:
    """Build the pinned marin-style command for the requested Python file scope."""

    command = (
        "uv",
        "run",
        "--frozen",
        "--script",
        "infra/marin-style.py",
        "--all-files" if all_files else "--changed-files",
    )
    return (*command, "--fix") if fix else command


def checks(*, fix: bool, all_files: bool) -> Sequence[Check]:
    """Build the ordered lint plan, applying Rust and Python formatters when requested."""

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
            "python style",
            marin_style_command(all_files=all_files, fix=fix),
            verification_command=(marin_style_command(all_files=all_files) if fix else None),
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
                "infra/marin-style.py",
            ),
        ),
        Check("unstaged whitespace", ("git", "diff", "--check")),
        Check("staged whitespace", ("git", "diff", "--cached", "--check")),
    )


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    """Parse file scope, formatting, and advisory review options."""

    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--all-files",
        action="store_true",
        help="check all tracked Python files and the complete Rust crate",
    )
    scope.add_argument(
        "--changed-files",
        action="store_true",
        help="check changed Python files; Rust checks still validate the complete crate",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply Rust and Python formatters before verifying other checks",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="run advisory review; following options pass through to pinned marin-style",
    )
    options, remaining = parser.parse_known_args(arguments)
    if remaining and not options.review:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    options.review_arguments = tuple(remaining)
    return options


def review_command(arguments: Sequence[str]) -> tuple[str, ...]:
    """Build the pinned, locally curated agentic review command."""

    return (
        "uv",
        "run",
        "--frozen",
        "--script",
        "infra/marin-style.py",
        "--review",
        *arguments,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the deterministic lint plan or the requested advisory review."""

    options = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # lint-review: allow ml-monolithic-function -- keep marin-style's documented consumer CLI.
    if options.review:
        return 0 if run_check(Check("agentic lint review", review_command(options.review_arguments))) else 1

    succeeded = True
    for check in checks(fix=options.fix, all_files=options.all_files):
        succeeded = run_check(check) and succeeded
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
