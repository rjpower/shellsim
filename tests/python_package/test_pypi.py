"""Offline package-loading checks cover staging, incompatible wheels, and failure isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import shellsim
import shellsim.pypi


def _fake_install(command: list[str], *, tag: str, native: bool = False) -> subprocess.CompletedProcess[str]:
    target = Path(command[command.index("--target") + 1])
    target.mkdir(parents=True)
    (target / "example.py").write_text("answer = 42\n")
    metadata = target / "example-1.0.dist-info"
    metadata.mkdir()
    (metadata / "WHEEL").write_text(f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n")
    if native:
        (target / "example.so").write_bytes(b"extension")
    return subprocess.CompletedProcess(command, 0, "", "")


def _add_numpy_wheel(command: list[str], version: str) -> None:
    target = Path(command[command.index("--target") + 1])
    package = target / "numpy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('host NumPy was staged')\n")
    (package / "_core.so").write_bytes(b"native")
    metadata = target / f"numpy-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: numpy\nVersion: {version}\n")
    (metadata / "WHEEL").write_text("Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp314-cp314-manylinux_x86_64\n")
    (metadata / "RECORD").write_text(
        f"numpy/__init__.py,,\nnumpy/_core.so,,\nnumpy-{version}.dist-info/METADATA,,\n"
        f"numpy-{version}.dist-info/WHEEL,,\nnumpy-{version}.dist-info/RECORD,,\n"
    )


def test_install_pypi_stages_pure_wheel_and_runs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = []

    def install(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append((command, kwargs))
        return _fake_install(command, tag="py3-none-any")

    monkeypatch.setattr(shellsim.pypi.subprocess, "run", install)
    environment = shellsim.Environment()
    environment.install_pypi("example==1.0")

    result = environment.run_python("from example import answer\nprint(answer)")
    assert result.stdout == b"42\n"
    assert result.returncode == 0
    assert observed[0][0][-1] == "example==1.0"
    assert observed[0][0][observed[0][0].index("--python") + 1] == "3.14"
    assert "--python-version" in observed[0][0]
    assert environment.run("pwd").stdout == b"/\n"


@pytest.mark.parametrize("version,accepted", [("2.5.3", True), ("2.5.2", False)])
def test_install_pypi_uses_only_matching_bundled_numpy(
    monkeypatch: pytest.MonkeyPatch, version: str, accepted: bool
) -> None:
    def install(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = _fake_install(command, tag="py3-none-any")
        _add_numpy_wheel(command, version)
        return result

    monkeypatch.setattr(shellsim.pypi.subprocess, "run", install)
    environment = shellsim.Environment()
    if accepted:
        environment.install_pypi("example")
        assert environment.run_python("import example; print(example.answer)").stdout == b"42\n"
        with pytest.raises(shellsim.SimulationError, match="No such file"):
            environment.read_file("/usr/lib/python3.14/site-packages/numpy/_core.so")
    else:
        with pytest.raises(shellsim.PackageInstallError, match="incompatible distributions"):
            environment.install_pypi("example")


@pytest.mark.parametrize("tag,native", [("cp314-cp314-manylinux_x86_64", False), ("py3-none-any", True)])
def test_install_pypi_rejects_native_artifacts_before_staging(
    monkeypatch: pytest.MonkeyPatch, tag: str, native: bool
) -> None:
    monkeypatch.setattr(
        shellsim.pypi.subprocess,
        "run",
        lambda command, **kwargs: _fake_install(command, tag=tag, native=native),
    )
    environment = shellsim.Environment()
    environment.write_file("/work/kept.py", "value = 1\n")

    with pytest.raises(shellsim.PackageInstallError, match="cannot stage incompatible distributions"):
        environment.install_pypi("example")

    assert environment.read_file("/work/kept.py") == b"value = 1\n"
    with pytest.raises(shellsim.SimulationError, match="No such file"):
        environment.read_file("/usr/lib/python3.14/site-packages/example.py")


def test_install_pypi_reports_resolution_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shellsim.pypi.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "No matching distribution"),
    )
    environment = shellsim.Environment()

    with pytest.raises(shellsim.PackageInstallError, match="No matching distribution"):
        environment.install_pypi("example")
    with pytest.raises(ValueError, match="PyPI distribution"):
        environment.install_pypi("https://example.test/package.whl")
    with pytest.raises(ValueError, match="PyPI distribution"):
        environment.install_pypi("--index-url=https://example.test")


def test_install_pypi_preserves_existing_vfs_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shellsim.pypi.subprocess,
        "run",
        lambda command, **kwargs: _fake_install(command, tag="py3-none-any"),
    )
    environment = shellsim.Environment()
    environment.mkdir("/usr/lib/python3.14/site-packages", parents=True)
    environment.write_file("/usr/lib/python3.14/site-packages/example.py", "answer = 7\n")

    with pytest.raises(shellsim.PackageInstallError, match="would overwrite"):
        environment.install_pypi("example")

    assert environment.read_file("/usr/lib/python3.14/site-packages/example.py") == b"answer = 7\n"
    with pytest.raises(shellsim.SimulationError, match="No such file"):
        environment.read_file("/usr/lib/python3.14/site-packages/example-1.0.dist-info/WHEEL")


def test_default_install_normalizes_modes_and_uses_atomic_mount(monkeypatch):
    def install(command, **kwargs):
        result = _fake_install(command, tag="py3-none-any")
        target = Path(command[command.index("--target") + 1])
        for path in target.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        script = target / "tool"
        script.write_text("#!/bin/sh\nprintf staged")
        script.chmod(0o700)
        return result

    monkeypatch.setattr(shellsim.pypi.subprocess, "run", install)
    environment = shellsim.Environment()

    def forbidden_preflight(path):
        raise AssertionError("package installer must check conflicts inside its atomic mount")

    monkeypatch.setattr(environment, "read_file", forbidden_preflight)
    environment.install_pypi("example")
    environment.install_pypi("example")
    assert environment.run("stat -c %a /usr/lib/python3.14/site-packages/example.py").stdout == b"644\n"
    assert environment.run("/usr/lib/python3.14/site-packages/tool").stdout == b"staged"


def test_default_install_preserves_resource_error(monkeypatch):
    def install(command, **kwargs):
        result = _fake_install(command, tag="py3-none-any")
        target = Path(command[command.index("--target") + 1])
        (target / "oversized.py").write_bytes(b"x" * 200_000)
        return result

    monkeypatch.setattr(shellsim.pypi.subprocess, "run", install)
    environment = shellsim.Environment(disk=100_000)
    with pytest.raises(shellsim.SimulationError):
        environment.install_pypi("example")
    assert environment.run("test ! -e /usr/lib/python3.14/site-packages/example.py").returncode == 0


def test_install_pypi_accepts_exact_bundled_distribution_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    def install(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _add_numpy_wheel(command, "2.5.3")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(shellsim.pypi.subprocess, "run", install)
    environment = shellsim.Environment()
    environment.install_pypi("numpy==2.5.3")

    result = environment.run_python("import numpy; print(numpy.__version__)")
    assert result.returncode == 0
    assert result.stdout == b"2.5.3\n"


@pytest.mark.parametrize("record_path", ["../outside", "example.py"])
def test_install_pypi_rejects_bundled_record_outside_its_files(
    monkeypatch: pytest.MonkeyPatch, record_path: str
) -> None:
    def install(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = _fake_install(command, tag="py3-none-any")
        _add_numpy_wheel(command, "2.5.3")
        target = Path(command[command.index("--target") + 1])
        record = target / "numpy-2.5.3.dist-info/RECORD"
        record.write_text(record.read_text().replace("numpy/__init__.py", record_path))
        return result

    monkeypatch.setattr(shellsim.pypi.subprocess, "run", install)
    environment = shellsim.Environment()
    with pytest.raises(shellsim.PackageInstallError, match="RECORD"):
        environment.install_pypi("example")
    with pytest.raises(shellsim.SimulationError, match="No such file"):
        environment.read_file("/usr/lib/python3.14/site-packages/example.py")


@pytest.mark.parametrize("name", ["pytest.py", "json.py", "scipy/__init__.py"])
def test_install_pypi_rejects_bundled_import_name_collisions(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    def install(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = _fake_install(command, tag="py3-none-any")
        path = Path(command[command.index("--target") + 1]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
        return result

    monkeypatch.setattr(shellsim.pypi.subprocess, "run", install)
    environment = shellsim.Environment()
    with pytest.raises(shellsim.PackageInstallError, match="conflict with bundled modules"):
        environment.install_pypi("example")
