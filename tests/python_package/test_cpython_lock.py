"""Check bounded lock inputs and real, opt-in WASI package selection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import shellsim
from shellsim._cpython_lock import _exported, _read_lock, _source_packages
from shellsim._cpython_universe import Universe
from shellsim.cpython import _requirements


def test_multi_spec_input_is_bounded_and_requires_named_distributions() -> None:
    assert _requirements(["packaging==26.3", "numpy==2.3.5"]) == ("packaging==26.3", "numpy==2.3.5")
    with pytest.raises(ValueError):
        _requirements([])
    with pytest.raises(ValueError):
        _requirements(["packaging"] * 257)
    with pytest.raises(ValueError):
        _requirements("https://example.invalid/wheel.whl")


def test_lock_rejects_local_dependency_and_requires_mounted_root() -> None:
    lock = {
        "package": [
            {"name": "project", "version": "0.1.0", "source": {"virtual": "."}},
            {"name": "helper", "version": "1.0", "source": {"path": "../helper"}},
        ]
    }
    with pytest.raises(shellsim.PackageInstallError, match="local"):
        _source_packages(lock, project_mounted=True)
    lock["package"].pop()
    with pytest.raises(shellsim.PackageInstallError, match="project_mounted"):
        _source_packages(lock, project_mounted=False)
    assert _source_packages(lock, project_mounted=True)[1] == set()


def test_lock_rejects_oversized_input_and_unpinned_export(tmp_path: Path) -> None:
    path = tmp_path / "uv.lock"
    path.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    with pytest.raises(shellsim.PackageInstallError, match="2 MiB"):
        _read_lock(path)
    exported = tmp_path / "requirements.txt"
    exported.write_text("https://example.invalid/pkg.whl\n")
    with pytest.raises(shellsim.PackageInstallError, match="unpinned"):
        _exported(exported, set())


def _live_runtime() -> tuple[shellsim.CPythonRuntime, shellsim.Environment]:
    bundle = os.environ.get("SHELLSIM_DYNAMIC_V2_BUNDLE")
    catalog = os.environ.get("SHELLSIM_CPYTHON_UNIVERSE")
    uv = os.environ.get("SHELLSIM_PATCHED_UV")
    if not all((bundle, catalog, uv)):
        pytest.skip("set SHELLSIM_DYNAMIC_V2_BUNDLE, SHELLSIM_CPYTHON_UNIVERSE, and SHELLSIM_PATCHED_UV")
    runtime = shellsim.CPythonRuntime(bundle, universe=catalog, uv=uv)
    environment = shellsim.Environment(cpu=4_000_000_000, memory=512 * 1024 * 1024, disk=128 * 1024 * 1024)
    runtime.mount(environment)
    return runtime, environment


def _lock_project(
    tmp_path: Path,
    runtime: shellsim.CPythonRuntime,
    dependencies: list[str],
    *,
    extra: str | None = None,
    group: str | None = None,
    wasi_only: bool = True,
) -> Path:
    assert runtime.universe is not None and runtime.uv is not None
    universe = Universe(runtime.universe, abi=runtime.manifest["dynamic_abi"], python_version=runtime.version)
    index = universe.index(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    metadata = (
        "[project]\nname = 'lock-test'\nversion = '0.1.0'\nrequires-python = '>=3.13'\n"
        + "dependencies = ["
        + ", ".join(repr(value) for value in dependencies)
        + "]\n"
    )
    if wasi_only:
        metadata += "\n[tool.uv]\nenvironments = [\"sys_platform == 'wasi'\"]\n"
    if extra is not None:
        metadata += f"\n[project.optional-dependencies]\npretty = [{extra!r}]\n"
    if group is not None:
        metadata += f"\n[dependency-groups]\ntest = [{group!r}]\n"
    (project / "pyproject.toml").write_text(metadata)
    completed = subprocess.run(
        [
            str(runtime.uv),
            "--no-config",
            "lock",
            "--offline",
            "--no-build",
            "--no-python-downloads",
            "--python",
            sys.executable,
            "--index",
            index.as_uri(),
            "--default-index",
            universe.default_index,
            "--index-strategy",
            "first-index",
        ],
        cwd=project,
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith(("UV_", "PIP_"))},
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return project / "uv.lock"


def test_multi_spec_resolution_and_conflict_are_atomic(tmp_path: Path) -> None:
    runtime, environment = _live_runtime()
    runtime.install_pypi(environment, ["packaging==26.3", "iniconfig==2.3.1"])
    result = runtime.run(environment, ["-c", "import packaging, iniconfig; print('ok')"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"ok\n"
    with pytest.raises(shellsim.PackageInstallError):
        runtime.install_pypi(environment, ["pygments==2.21.0", "pygments==2.20.0"])
    with pytest.raises(shellsim.SimulationError):
        environment.read_file(runtime.site_packages + "/pygments/__init__.py")


def test_lock_reselects_guest_markers_and_native_wheel(tmp_path: Path) -> None:
    runtime, environment = _live_runtime()
    lock = _lock_project(
        tmp_path,
        runtime,
        ["numpy==2.3.5", "packaging==26.3; sys_platform == 'wasi'", "iniconfig==2.3.1; sys_platform == 'linux'"],
    )
    runtime.install_lock(environment, lock, project_mounted=True)
    result = runtime.run(environment, ["-c", "import numpy, packaging; print(numpy.arange(4).sum())"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"6\n"
    with pytest.raises(shellsim.SimulationError):
        environment.read_file(runtime.site_packages + "/iniconfig/__init__.py")


def test_lock_installs_pure_pytest_closure(tmp_path: Path) -> None:
    runtime, environment = _live_runtime()
    lock = _lock_project(tmp_path, runtime, ["pytest==8.4.1"])
    runtime.install_lock(environment, lock, project_mounted=True)
    result = runtime.run(
        environment, ["-c", "import pytest, pluggy, packaging, iniconfig, pygments; print(pytest.__version__)"]
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"8.4.1\n"


def test_universal_host_lock_uses_guest_marker_branch(tmp_path: Path) -> None:
    runtime, environment = _live_runtime()
    lock = _lock_project(
        tmp_path,
        runtime,
        ["packaging==26.3; sys_platform == 'wasi'", "iniconfig==2.3.1; sys_platform == 'linux'"],
        wasi_only=False,
    )
    assert "supported-markers" not in lock.read_text()
    runtime.install_lock(environment, lock, project_mounted=True)
    result = runtime.run(environment, ["-c", "import packaging; print('wasi')"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"wasi\n"
    with pytest.raises(shellsim.SimulationError):
        environment.read_file(runtime.site_packages + "/iniconfig/__init__.py")


def test_lock_unavailable_native_pin_keeps_vfs_intact(tmp_path: Path) -> None:
    runtime, environment = _live_runtime()
    lock = _lock_project(tmp_path, runtime, ["numpy==2.3.5"])
    data = lock.read_text().replace("2.3.5", "2.2.0")
    lock.write_text(data)
    with pytest.raises(shellsim.PackageInstallError, match="curated WASI versions available"):
        runtime.install_lock(environment, lock, project_mounted=True)
    with pytest.raises(shellsim.SimulationError):
        environment.read_file(runtime.site_packages + "/numpy/__init__.py")


def test_lock_selects_extras_and_groups_explicitly(tmp_path: Path) -> None:
    runtime, environment = _live_runtime()
    lock = _lock_project(
        tmp_path,
        runtime,
        ["packaging==26.3"],
        extra="pygments==2.21.0",
        group="iniconfig==2.3.1",
    )
    with pytest.raises(shellsim.PackageInstallError, match="no group"):
        runtime.install_lock(environment, lock, groups=("missing",), project_mounted=True)
    with pytest.raises(shellsim.SimulationError):
        environment.read_file(runtime.site_packages + "/packaging/__init__.py")
    runtime.install_lock(environment, lock, extras=("pretty",), groups=("test",), project_mounted=True)
    result = runtime.run(environment, ["-c", "import packaging, pygments, iniconfig; print('selected')"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"selected\n"


def test_dependency_free_lock_is_valid_noop(tmp_path: Path) -> None:
    runtime, environment = _live_runtime()
    lock = _lock_project(tmp_path, runtime, [])
    runtime.install_lock(environment, lock, project_mounted=True)
    result = runtime.run(environment, ["-c", "import sys; print(sys.platform)"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"wasi\n"


@pytest.mark.parametrize("dependencies", [[], ["packaging==26.3"]])
def test_lock_incompatible_python_range_is_rejected_before_staging(tmp_path: Path, dependencies: list[str]) -> None:
    runtime, environment = _live_runtime()
    lock = _lock_project(tmp_path, runtime, dependencies)
    lock.write_text(lock.read_text().replace('requires-python = ">=3.13"', 'requires-python = ">=3.14"'))
    with pytest.raises(shellsim.PackageInstallError, match="WASI guest is 3.13.7"):
        runtime.install_lock(environment, lock, project_mounted=True)
    with pytest.raises(shellsim.SimulationError):
        environment.read_file(runtime.site_packages + "/packaging/__init__.py")


def test_lock_rejects_unsupported_guest_environment(tmp_path: Path) -> None:
    runtime, environment = _live_runtime()
    lock = _lock_project(tmp_path, runtime, [])
    lock.write_text(lock.read_text().replace("sys_platform == 'wasi'", "sys_platform == 'linux'"))
    with pytest.raises(shellsim.PackageInstallError, match="exclude the WASI guest"):
        runtime.install_lock(environment, lock, project_mounted=True)
