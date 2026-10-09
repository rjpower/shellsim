"""Run one unchanged TaskTrove verifier inside the mounted CPython/WASI guest.

This is opt-in because the upstream task archive, WASI bundle, and offline wheel
catalog are external build artifacts. The candidate runs only inside shellsim.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import shellsim


def test_calibforge_candidate_and_original_pytest_in_guest() -> None:
    bundle = os.environ.get("SHELLSIM_TASK_CPYTHON_BUNDLE")
    universe = os.environ.get("SHELLSIM_TASK_UNIVERSE")
    uv = os.environ.get("SHELLSIM_PATCHED_UV")
    tasks = os.environ.get("SHELLSIM_TASKTROVE_TASKS", "/tmp/tasktrove-seven-native")
    if not all((bundle, universe, uv)):
        pytest.skip("set SHELLSIM_TASK_CPYTHON_BUNDLE, SHELLSIM_TASK_UNIVERSE, and SHELLSIM_PATCHED_UV")

    task = Path(tasks) / "calibforge-fb1e75441a94.parquet"
    assert task.is_dir(), f"local CalibForge task archive is unavailable: {task}"
    runtime = shellsim.CPythonRuntime(bundle, universe=universe, uv=uv, venv="/app/.venv")
    environment = shellsim.Environment(cpu=20_000_000_000, memory=512 * 1024 * 1024, disk=128 * 1024 * 1024)
    runtime.mount(environment)
    environment.mount(task / "environment/files", "/app/data")
    environment.mount(task / "tests", "/tests")
    candidate = Path(__file__).resolve().parents[1] / "fixtures/tasktrove_calibforge/reconcile.py"
    environment.write_file("/app/reconcile.py", candidate.read_bytes())
    runtime.install_pypi(environment, "pytest==8.4.1")

    empty = environment.run("cd /app; python -m pytest -q /tests/test_outputs.py")
    assert empty.returncode == 1, empty.stdout + empty.stderr
    assert b"test_reconciliation_report_exists" in empty.stdout + empty.stderr
    assert b"failed" in empty.stdout + empty.stderr

    candidate_run = environment.run("cd /app; python reconcile.py")
    assert candidate_run.returncode == 0, candidate_run.stderr
    verified = environment.run("cd /app; pytest -q /tests/test_outputs.py")
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert b"17 passed" in verified.stdout
    print(verified.stdout.decode())
