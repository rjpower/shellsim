"""Run an original TaskTrove verifier against C compiled inside the guest.

The unchanged task requires a Python/C++/Java entry point. A small CPython
launcher forwards its streams to the C binary through the virtual process API.
The host only provisions verified compiler data, task files and requirements.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import shellsim


@pytest.fixture
def c_task_environment() -> shellsim.Environment:
    catalog = os.environ.get("SHELLSIM_GUEST_CATALOG")
    tasks = os.environ.get("SHELLSIM_CODEELO_TASKS")
    if not catalog or not tasks:
        pytest.skip("set SHELLSIM_GUEST_CATALOG and SHELLSIM_CODEELO_TASKS for Clang and task acceptance")
    environment = shellsim.Context(
        pypi_catalog=catalog,
        trusted_precompiled=os.environ.get("SHELLSIM_GUEST_TRUSTED_PRECOMPILED") == "1",
    ).Environment(
        pypi_packages=["pytest"], tools=["clang", "make"],
        cpu=500_000_000_000, memory=8 * 1024**3, disk=768 * 1024**2,
    )
    environment.mount(Path(tasks) / "codeelo-0000/tests", "/tests")
    environment.mkdir("/app", parents=True)
    candidate = Path(__file__).resolve().parents[1] / "fixtures/tasktrove_c_build"
    for name in ("brackets.c", "Makefile", "solution.py"):
        environment.write_file("/app/" + name, (candidate / name).read_bytes())
    return environment


def test_guest_compiled_c_candidate_passes_original_verifier(
    c_task_environment: shellsim.Environment
) -> None:
    environment = c_task_environment
    build = "cd /app && make -j2"
    compiled = environment.run(build)
    assert compiled.returncode == 0, compiled.stderr
    original_binary = environment.read_file("/app/solver")
    assert original_binary.startswith(b"\0asm")
    current = environment.run("cd /app && make -q")
    assert current.returncode == 0, current.stdout + current.stderr

    result = environment.run("bash /tests/test.sh")
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    assert b"Results: 20/20 passed" in result.stdout
    assert environment.read_file("/logs/verifier/reward.txt").strip() == b"1"
    verified = environment.run("pytest -q /tests/test_state.py")
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert b"1 passed" in verified.stdout

    # Separate the edit from the last build using guest time, so make must
    # discover a newer prerequisite without a forced-rebuild option.
    elapsed = environment.run("sleep 1")
    assert elapsed.returncode == 0, elapsed.stderr
    environment.write_file("/app/brackets.c", '#include <stdio.h>\nint main(void) { puts("incorrect"); return 0; }\n')
    stale = environment.run("cd /app && make -q")
    assert stale.returncode == 1, stale.stdout + stale.stderr
    rebuilt = environment.run(build)
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert environment.read_file("/app/solver") != original_binary
    current = environment.run("cd /app && make -q")
    assert current.returncode == 0, current.stdout + current.stderr
    rejected = environment.run("bash /tests/test.sh")
    assert rejected.returncode == 0, rejected.stderr
    assert rejected.stderr == b""
    assert b"Results: 0/20 passed" in rejected.stdout
    assert environment.read_file("/logs/verifier/reward.txt").strip() == b"0"
    failed = environment.run("pytest -q /tests/test_state.py")
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert b"1 failed" in failed.stdout
