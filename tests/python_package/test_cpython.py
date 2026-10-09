"""Bundle integrity and package staging are offline; a built guest is opt-in."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest
import shellsim
import shellsim.cpython


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "rootfs"
    binary = root / "usr/bin/python3.wasm"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\0asm\x01\0\0\0")
    binary.chmod(0o755)
    manifest = {
        "recipe": {"version": "3.13.7", "target": "wasm32-wasip1", "prefix": "/usr"},
        "site_packages": "/usr/lib/python3.13/site-packages",
        "files": {"/usr/bin/python3.wasm": hashlib.sha256(binary.read_bytes()).hexdigest()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def provider_metadata(bundle: Path, manifest: dict, name: str, version: str) -> str:
    """Add the exact seeded provider metadata to a verified test bundle."""
    path = f"/usr/lib/python3.13/site-packages/{name}-{version}.dist-info"
    source = bundle / "rootfs" / path.lstrip("/")
    source.mkdir(parents=True)
    files = {
        "METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        "WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py313-none-any\n",
        "RECORD": "",
    }
    for filename, content in files.items():
        (source / filename).write_text(content)
        manifest["files"][path + "/" + filename] = hashlib.sha256(content.encode()).hexdigest()
    return path


def test_bundle_integrity_checked_before_mount(bundle: Path) -> None:
    runtime = shellsim.CPythonRuntime(bundle)
    environment = shellsim.Environment()
    (bundle / "rootfs/usr/bin/python3.wasm").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        runtime.mount(environment)
    assert environment.run("test -e /usr/bin/python3.wasm").returncode != 0


def test_bundle_rejects_other_target(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["recipe"]["target"] = "x86_64-linux"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        shellsim.CPythonRuntime(bundle)


@pytest.mark.parametrize(
    ("target", "abi", "accepted"),
    [
        ("wasm32-wasip1", "shellsim-wasi-sdk34-cpython3137-v2", True),
        ("wasm32-wasip1-threads", "shellsim-wasi-sdk34-cpython3137-threads-v3", True),
        ("wasm32-wasip1", "shellsim-wasi-sdk34-cpython3137-threads-v3", False),
        ("wasm32-wasip1-threads", "shellsim-wasi-sdk34-cpython3137-v2", False),
        ("wasm32-wasip1-threads", None, False),
        ("wasm32-wasip1", "unknown-dynamic-abi", False),
    ],
)
def test_bundle_admits_only_matched_runtime_profiles(
    bundle: Path, target: str, abi: str | None, accepted: bool
) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["recipe"]["target"] = target
    if abi is not None:
        manifest["dynamic_abi"] = abi
        manifest["recipe"]["dynamic_abi"] = abi
    manifest_path.write_text(json.dumps(manifest))
    if accepted:
        assert shellsim.CPythonRuntime(bundle).manifest["dynamic_abi"] == abi
    else:
        with pytest.raises(ValueError):
            shellsim.CPythonRuntime(bundle)


def test_bundle_rejects_recipe_abi_disagreement(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dynamic_abi"] = "shellsim-wasi-sdk34-cpython3137-v2"
    manifest["recipe"]["dynamic_abi"] = "shellsim-wasi-sdk34-cpython3137-threads-v3"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        shellsim.CPythonRuntime(bundle)


def test_runtime_manifest_read_is_bounded_before_json_decode(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(manifest + b" " * (1024 * 1024))
    shellsim.CPythonRuntime(bundle)

    manifest_path.write_bytes(b"invalid JSON" + b" " * (4 * 1024 * 1024))
    with pytest.raises(ValueError, match="exceeds 4 MiB"):
        shellsim.CPythonRuntime(bundle)


@pytest.mark.parametrize("native", [False, True])
def test_local_wheel_staging(bundle: Path, tmp_path: Path, native: bool) -> None:
    wheel = tmp_path / "example.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("example.py", "answer = 42\n")
        archive.writestr("example-1.0.dist-info/WHEEL", "Root-Is-Purelib: true\nTag: py3-none-any\n")
        if native:
            archive.writestr("example.so", b"native")
    environment = shellsim.Environment()
    runtime = shellsim.CPythonRuntime(bundle)
    if native:
        with pytest.raises(shellsim.PackageInstallError):
            runtime.install_wheel(environment, wheel)
        assert environment.run("test -e /usr/lib/python3.13/site-packages/example.py").returncode != 0
        return
    runtime.install_wheel(environment, wheel)
    assert environment.read_file(runtime.site_packages + "/example.py") == b"answer = 42\n"
    environment.write_file(runtime.site_packages + "/example.py", "user data")
    with pytest.raises(shellsim.PackageInstallError):
        runtime.install_wheel(environment, wheel)
    assert environment.read_file(runtime.site_packages + "/example.py") == b"user data"


def test_pypi_resolves_for_guest_and_does_not_substitute_modules(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def install(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[command.index("--python-version") + 1] == "3.13"
        assert "--only-binary" not in command
        target = Path(command[command.index("--target") + 1])
        target.mkdir(exist_ok=True)
        (target / "numpy.py").write_text("answer = 42\n")
        metadata = target / "numpy-2.5.3.dist-info"
        metadata.mkdir()
        (metadata / "WHEEL").write_text("Root-Is-Purelib: true\nTag: py313-none-any\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(shellsim.cpython.subprocess, "run", install)
    runtime = shellsim.CPythonRuntime(bundle)
    environment = shellsim.Environment()
    runtime.install_pypi(environment, "numpy==2.5.3")
    assert environment.read_file(runtime.site_packages + "/numpy.py") == b"answer = 42\n"


def test_source_built_guest() -> None:
    path = os.environ.get("SHELLSIM_CPYTHON_BUNDLE")
    if path is None:
        pytest.skip("set SHELLSIM_CPYTHON_BUNDLE to a source-built CPython WASI bundle")
    runtime = shellsim.CPythonRuntime(path)
    environment = shellsim.Environment(cpu=2_000_000_000, memory=256 * 1024 * 1024, disk=64 * 1024 * 1024)
    runtime.mount(environment)
    result = runtime.run(environment, ["-c", "import json, sys; print(sys.platform); print(json.dumps([1, 2]))"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"wasi\n[1, 2]\n"


def test_native_provider_seeds_resolution(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["builtin_modules"] = ["pycosat"]
    metadata_path = provider_metadata(bundle, manifest, "pycosat", "0.6.6")
    manifest["native_ports"] = [
        {
            "name": "pycosat",
            "version": "0.6.6",
            "builtin_modules": ["pycosat"],
            "requires_dist": [],
            "dist_info": metadata_path,
        }
    ]
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    def resolve(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        target = Path(command[command.index("--target") + 1])
        metadata = (target / "pycosat-0.6.6.dist-info/METADATA").read_text()
        assert "Name: pycosat\nVersion: 0.6.6\n" in metadata
        constraint = Path(command[command.index("--constraint") + 1])
        assert constraint.read_text() == "pycosat==0.6.6\n"
        assert command[-1] == "pycosat==0.6.6"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(shellsim.cpython.subprocess, "run", resolve)
    runtime = shellsim.CPythonRuntime(bundle)
    environment = shellsim.Environment()
    runtime.install_pypi(environment, "pycosat==0.6.6")
    assert b"Version: 0.6.6\n" in environment.read_file(runtime.site_packages + "/pycosat-0.6.6.dist-info/METADATA")


@pytest.mark.parametrize("modules", [None, [], ["numpy._core.missing"], ["numpy.._core"]])
def test_native_provider_rejects_missing_or_unlinked_modules(
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    modules: object,
) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["builtin_modules"] = ["numpy._core._multiarray_umath"]
    manifest["native_ports"] = [{"name": "numpy", "version": "2.3.5", "requires_dist": [], "builtin_modules": modules}]
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid provider must fail before host resolution")

    monkeypatch.setattr(shellsim.cpython.subprocess, "run", forbidden)
    runtime = shellsim.CPythonRuntime(bundle)
    with pytest.raises(shellsim.PackageInstallError, match="invalid builtin native provider"):
        runtime.install_pypi(shellsim.Environment(), "numpy")


def test_native_distribution_can_have_multiple_qualified_builtin_modules(
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    modules = ["numpy._core._multiarray_umath", "numpy.fft._pocketfft_umath"]
    manifest["builtin_modules"] = modules
    metadata_path = provider_metadata(bundle, manifest, "numpy", "2.3.5")
    manifest["native_ports"] = [
        {
            "name": "numpy",
            "version": "2.3.5",
            "requires_dist": [],
            "builtin_modules": modules,
            "dist_info": metadata_path,
        }
    ]
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    def resolve(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        target = Path(command[command.index("--target") + 1])
        assert "Name: numpy\nVersion: 2.3.5\n" in (target / "numpy-2.3.5.dist-info/METADATA").read_text()
        assert Path(command[command.index("--constraint") + 1]).read_text() == "numpy==2.3.5\n"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(shellsim.cpython.subprocess, "run", resolve)
    runtime = shellsim.CPythonRuntime(bundle)
    environment = shellsim.Environment()
    runtime.install_pypi(environment, "numpy==2.3.5")
    assert b"Version: 2.3.5\n" in environment.read_file(runtime.site_packages + "/numpy-2.3.5.dist-info/METADATA")
