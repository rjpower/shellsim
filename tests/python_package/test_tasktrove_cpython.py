"""Opt-in task acceptance executes both candidate and original verifier in the guest.

Task archives remain external fixtures. The supplied codeelo-0000/codeelo-0002
directories must contain the unchanged TaskTrove verifier and twenty input pairs.
The local universe must include the pytest closure so replay needs no network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import shellsim

BRACKETS = """import sys

def balanced(text):
    depth = 0
    for char in text:
        depth += 1 if char == '(' else -1
        if depth < 0:
            return False
    return depth == 0

lines = sys.stdin.read().split()
for text in lines[1:1 + int(lines[0])]:
    print('YES' if balanced(text) else 'NO')
"""

GCD = """import math
import sys

values = iter(sys.stdin.read().split())
count = int(next(values))
for _ in range(count):
    print(math.gcd(int(next(values)), int(next(values))))
"""


@pytest.fixture
def task_runtime() -> tuple[shellsim.CPythonRuntime, Path]:
    keys = (
        "SHELLSIM_TASK_CPYTHON_BUNDLE",
        "SHELLSIM_TASK_UNIVERSE",
        "SHELLSIM_PATCHED_UV",
        "SHELLSIM_CODEELO_TASKS",
    )
    if any(not os.environ.get(key) for key in keys):
        pytest.skip("set the task runtime, offline universe, resolver, and Codeelo fixture paths")
    return (
        shellsim.CPythonRuntime(
            os.environ[keys[0]],
            universe=os.environ[keys[1]],
            uv=os.environ[keys[2]],
        ),
        Path(os.environ[keys[3]]),
    )


@pytest.mark.parametrize(
    "task,candidate",
    [("codeelo-0000", BRACKETS), ("codeelo-0002", GCD)],
    ids=["brackets", "gcd"],
)
def test_original_task_verifier_accepts_solution_and_rejects_wrong_output(
    task_runtime: tuple[shellsim.CPythonRuntime, Path], task: str, candidate: str
) -> None:
    runtime, fixtures = task_runtime
    environment = shellsim.Environment(cpu=50_000_000_000, memory=1024**3, disk=256 * 1024**2)
    runtime.mount(environment)
    runtime.install_pypi(environment, "pytest==8.4.1")
    environment.mount(fixtures / task / "tests", "/tests")
    environment.mkdir("/app", parents=True)
    environment.write_file("/app/solution.py", candidate)
    identity = environment.run("python -c 'import sys; print(sys.platform); print(sys.prefix)'")
    assert identity.returncode == 0, identity.stderr
    assert identity.stdout == b"wasi\n/work/.venv\n"

    result = environment.run("bash /tests/test.sh")
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    assert b"Results: 20/20 passed" in result.stdout
    assert environment.read_file("/logs/verifier/reward.txt").strip() == b"1"
    # The upstream wrapper suppresses pytest's exit; check it independently.
    verified = environment.run("python -m pytest -q /tests/test_state.py")
    assert verified.returncode == 0, verified.stderr + verified.stdout
    assert b"1 passed" in verified.stdout

    environment.write_file("/app/solution.py", "print('incorrect')\n")
    rejected = environment.run("bash /tests/test.sh")
    assert rejected.returncode == 0, rejected.stderr
    assert rejected.stderr == b""
    assert b"Results: 0/20 passed" in rejected.stdout
    assert environment.read_file("/logs/verifier/reward.txt").strip() == b"0"
    failed = environment.run("pytest -q /tests/test_state.py")
    assert failed.returncode == 1, failed.stderr + failed.stdout
    assert b"1 failed" in failed.stdout
