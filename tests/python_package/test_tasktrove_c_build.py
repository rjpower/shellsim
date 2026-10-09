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
import shellsim_c_toolchain
from shellsim.native_packages import _NativePackageUniverse


@pytest.fixture(params=("compiler", "native-packages"))
def c_build_mode(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def c_task_environment(c_build_mode: str) -> shellsim.Environment:
    keys = (
        "SHELLSIM_TASK_CPYTHON_BUNDLE",
        "SHELLSIM_TASK_UNIVERSE",
        "SHELLSIM_PATCHED_UV",
        "SHELLSIM_CODEELO_TASKS",
    )
    if any(not os.environ.get(key) for key in keys):
        pytest.skip("set the task runtime, offline universe, resolver, and Codeelo fixture paths")
    catalog = os.environ.get("SHELLSIM_NATIVE_CATALOG")
    if c_build_mode == "native-packages" and catalog is None:
        pytest.skip("set SHELLSIM_NATIVE_CATALOG to the verified GNU make and C compiler catalog")
    runtime = shellsim.CPythonRuntime(os.environ[keys[0]], universe=os.environ[keys[1]], uv=os.environ[keys[2]])
    environment = shellsim.Environment(cpu=50_000_000_000, memory=1024**3, disk=256 * 1024**2)
    runtime.mount(environment)
    runtime.install_pypi(environment, "pytest==8.4.1")
    if c_build_mode == "native-packages":
        selected = _NativePackageUniverse(catalog).install(
            environment, ["make>=4.4,<5", "shellsim-c-toolchain==0.1.30"]
        )
        assert {"make", "shellsim-c-toolchain"} <= selected.keys()
        assert environment.read_file("/usr/bin/make").startswith(b"\0asm")
    else:
        shellsim_c_toolchain.install_c_toolchain(environment)
    environment.mount(Path(os.environ[keys[3]]) / "codeelo-0000/tests", "/tests")
    environment.mkdir("/app", parents=True)
    candidate = Path(__file__).resolve().parents[1] / "fixtures/tasktrove_c_build"
    for name in ("brackets.c", "Makefile", "solution.py"):
        environment.write_file("/app/" + name, (candidate / name).read_bytes())
    return environment


def test_guest_compiled_c_candidate_passes_original_verifier(
    c_task_environment: shellsim.Environment, c_build_mode: str
) -> None:
    environment = c_task_environment
    build = (
        "cd /app && make -j2"
        if c_build_mode == "native-packages"
        else "cd /app && cc -O2 -o solver brackets.c && chmod +x solver"
    )
    compiled = environment.run(build)
    assert compiled.returncode == 0, compiled.stderr
    original_binary = environment.read_file("/app/solver")
    assert original_binary.startswith(b"\0asm")
    if c_build_mode == "native-packages":
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

    if c_build_mode == "native-packages":
        # Separate the edit from the last build using guest time, so make must
        # discover a newer prerequisite without a forced-rebuild option.
        elapsed = environment.run("sleep 1")
        assert elapsed.returncode == 0, elapsed.stderr
    environment.write_file("/app/brackets.c", '#include <stdio.h>\nint main(void) { puts("incorrect"); return 0; }\n')
    if c_build_mode == "native-packages":
        stale = environment.run("cd /app && make -q")
        assert stale.returncode == 1, stale.stdout + stale.stderr
    rebuilt = environment.run(build)
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert environment.read_file("/app/solver") != original_binary
    if c_build_mode == "native-packages":
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
