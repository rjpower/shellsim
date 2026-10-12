"""Check release descriptor setup and public catalog Python/tool integration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import shellsim
from shellsim._cpython_universe import Universe


@pytest.mark.parametrize(
    "setup_args",
    [
        {"pypi": "pytest==8.4.1", "lock": "uv.lock", "project": "project"},
        {"lock": "uv.lock"},
        {"extras": ("pretty",)},
        {"pypi": []},
        {"tools": [123]},
        {"tools": "https://example.invalid/tool"},
    ],
)
def test_invalid_setup_request_fails_before_release_fetch(monkeypatch: pytest.MonkeyPatch, setup_args: dict) -> None:
    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("invalid setup fetched a release")

    monkeypatch.setattr(shellsim.CPythonRuntime, "from_release", unexpected_fetch)
    with pytest.raises((TypeError, ValueError)):
        shellsim.Context().from_release("missing-release.json", **setup_args)


def test_project_venv_is_rejected_before_release_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    (project / ".venv/bin").mkdir(parents=True)
    (project / ".venv/bin/python").write_bytes(b"source interpreter")

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("overlapping project fetched a release")

    monkeypatch.setattr(shellsim.CPythonRuntime, "from_release", unexpected_fetch)
    with pytest.raises(ValueError, match="overlap"):
        shellsim.Context().from_release("missing-release.json", project=project)


def _descriptor() -> Path:
    value = os.environ.get("SHELLSIM_RELEASE_DESCRIPTOR")
    if value is None:
        pytest.skip("set SHELLSIM_RELEASE_DESCRIPTOR to an accepted Python and native release")
    return Path(value)


def test_release_factory_defaults_run_guest_python(tmp_path: Path) -> None:
    env = shellsim.Context(cache_dir=tmp_path / "cache").from_release(_descriptor(), pypi="packaging==26.3")
    result = env.run("python -c 'import packaging; print(packaging.__version__)'")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"26.3\n"


def test_catalog_installs_python_and_native_tools() -> None:
    catalog = os.environ.get("SHELLSIM_GUEST_CATALOG")
    if not catalog:
        pytest.skip("set SHELLSIM_GUEST_CATALOG to an exported Python, Clang and make catalog")
    env = shellsim.Context(
        pypi_catalog=catalog,
        trusted_precompiled=os.environ.get("SHELLSIM_GUEST_TRUSTED_PRECOMPILED") == "1",
    ).Environment(
        pypi_packages=["pytest", "numpy"], tools=["clang", "make"],
        cpu=500_000_000_000, memory=8 * 1024**3, disk=768 * 1024**2,
    )
    result = env.run_python(
        "import pytest, numpy\n"
        "with pytest.raises(ValueError):\n"
        "    numpy.arange(4).reshape(3)\n"
        "print(numpy.arange(4).sum())"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"6\n"
    env.write_file("/work/main.c", '#include <stdio.h>\nint main(void) { puts("guest C"); return 0; }\n')
    env.write_file("/work/Makefile", "all: hello.wasm\nhello.wasm: main.c\n\tcc $< -o $@\n\tchmod +x $@\n")
    built = env.run("cd /work && make -j2 && make -q && ./hello.wasm")
    assert built.returncode == 0, built.stderr
    assert built.stdout.endswith(b"guest C\n")


def test_release_factory_installs_selected_uv_lock(tmp_path: Path) -> None:
    descriptor = _descriptor()
    runtime = shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache")
    assert runtime.universe is not None and runtime.uv is not None
    catalog = Universe(runtime.universe, abi=runtime.manifest["dynamic_abi"], python_version=runtime.version)
    index = catalog.index(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname = 'factory-lock'\nversion = '0.1.0'\nrequires-python = '>=3.13'\n"
        "dependencies = [\"packaging==26.3; sys_platform == 'wasi'\", "
        "\"iniconfig==2.3.1; sys_platform == 'linux'\"]\n"
    )
    locked = subprocess.run(
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
            catalog.default_index,
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
    assert locked.returncode == 0, locked.stderr

    env = shellsim.Context(cache_dir=tmp_path / "cache").from_release(descriptor, lock=project / "uv.lock", project=project, limits=shellsim.Limits(cpu=4_000_000_000, memory=512 * 1024**2, disk=256 * 1024**2))
    assert env.read_file("/work/pyproject.toml") == (project / "pyproject.toml").read_bytes()
    result = env.run("python -c 'import packaging; print(packaging.__version__)'")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"26.3\n"
    with pytest.raises(shellsim.SimulationError):
        env.read_file("/work/.venv/lib/python3.13/site-packages/iniconfig/__init__.py")
