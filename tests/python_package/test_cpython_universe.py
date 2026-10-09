"""Check ABI-scoped catalog rejection and opt-in public CPython package workflows."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
import shellsim
from shellsim._cpython_universe import (
    Universe,
    _add_wheel_to_set,
    _inspect_wheel,
    _stage_console_scripts,
    _stage_provider,
    _verify_file,
    _verify_wasm,
    _WheelInspection,
)

ABI = "shellsim-wasi-sdk34-cpython3137-v2"


def make_catalog(root: Path, **updates: object) -> Path:
    root.mkdir()
    catalog = {
        "schema_version": 1,
        "abi": ABI,
        "target": "wasm32-wasip1",
        "python_version": "3.13.7",
        "packages": [],
        "native_providers": [],
    }
    catalog.update(updates)
    (root / "catalog.json").write_text(json.dumps(catalog))
    return root


def make_bundle(root: Path) -> Path:
    binary = root / "rootfs/usr/bin/python3.wasm"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\0asm\x01\0\0\0")
    binary.chmod(0o755)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "recipe": {"version": "3.13.7", "target": "wasm32-wasip1", "prefix": "/usr"},
                "site_packages": "/usr/lib/python3.13/site-packages",
                "dynamic_abi": ABI,
                "files": {"/usr/bin/python3.wasm": hashlib.sha256(binary.read_bytes()).hexdigest()},
            }
        )
    )
    return root


def marked_wasm(abi: str = ABI) -> bytes:
    payload = b"\x0cshellsim.abi" + abi.encode()
    assert len(payload) < 128
    return b"\0asm\x01\0\0\0" + bytes((0, len(payload))) + payload


def test_dynamic_bundle_requires_explicit_universe(tmp_path: Path) -> None:
    runtime = shellsim.CPythonRuntime(make_bundle(tmp_path / "bundle"))
    with pytest.raises(shellsim.PackageInstallError, match="local universe"):
        runtime.install_pypi(shellsim.Environment(), "numpy==2.3.5")


def test_dynamic_mount_selects_real_venv_launchers(tmp_path: Path) -> None:
    runtime = shellsim.CPythonRuntime(make_bundle(tmp_path / "bundle"))
    environment = shellsim.Environment()
    runtime.mount(environment)
    assert environment.read_file("/work/.venv/pyvenv.cfg").startswith(b"home = /usr/bin\n")
    assert environment.run("readlink /usr/bin/python3; readlink /work/.venv/bin/python").stdout == (
        b"/usr/bin/python3.wasm\n/usr/bin/python3.wasm\n"
    )
    assert environment.run("python3.14 -c 'print(42)'").stdout == b"42\n"
    assert environment.run('printf \'%s\\n\' "$PATH" "$VIRTUAL_ENV"').stdout == (
        b"/work/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n/work/.venv\n"
    )
    runtime.mount(environment)
    assert environment.run("printf '%s\\n' \"$PATH\"").stdout.count(b"/work/.venv/bin") == 1
    with pytest.raises(shellsim.SimulationError, match="different CPython runtime"):
        shellsim.CPythonRuntime(runtime.bundle).mount(environment)


def test_mount_preserves_custom_path_and_failed_mount_environment(tmp_path: Path) -> None:
    runtime = shellsim.CPythonRuntime(make_bundle(tmp_path / "bundle"))
    environment = shellsim.Environment()
    environment.run("export PATH=/custom/bin:/usr/bin:/bin")
    runtime.mount(environment)
    assert environment.run('printf \'%s\\n\' "$PATH" "$VIRTUAL_ENV"').stdout == (
        b"/work/.venv/bin:/custom/bin:/usr/bin:/bin\n/work/.venv\n"
    )

    empty_path = shellsim.Environment()
    empty_path.run("export PATH=''")
    runtime.mount(empty_path)
    assert empty_path.run("printf '%s\\n' \"$PATH\"").stdout == b"/work/.venv/bin:\n"
    runtime.mount(empty_path)
    assert empty_path.run("printf '%s\\n' \"$PATH\"").stdout == b"/work/.venv/bin:\n"

    conflicting = shellsim.Environment()
    conflicting.run("export PATH=/custom/bin:/usr/bin:/bin")
    conflicting.write_file("/usr/bin/python", b"custom launcher")
    with pytest.raises(shellsim.SimulationError, match="conflicts"):
        runtime.mount(conflicting)
    assert conflicting.run('printf \'%s\\n\' "$PATH" "${VIRTUAL_ENV-unset}"').stdout == (
        b"/custom/bin:/usr/bin:/bin\nunset\n"
    )
    with pytest.raises(shellsim.SimulationError, match="mounted CPython"):
        conflicting.install_lock("/tmp/uv.lock", project_mounted=True)


def test_dynamic_mount_rolls_back_launcher_conflict(tmp_path: Path) -> None:
    runtime = shellsim.CPythonRuntime(make_bundle(tmp_path / "bundle"))
    environment = shellsim.Environment()
    environment.write_file("/usr/bin/python", b"custom launcher")
    with pytest.raises(shellsim.SimulationError, match="conflicts"):
        runtime.mount(environment)
    assert environment.read_file("/usr/bin/python") == b"custom launcher"
    with pytest.raises(shellsim.SimulationError):
        environment.read_file("/usr/bin/python3.wasm")


def test_console_script_relocation_uses_guest_venv_shebang(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    site = staging / "work/.venv/lib/python3.13/site-packages"
    scripts = site / "bin"
    scripts.mkdir(parents=True)
    script = scripts / "sample-tool"
    script.write_bytes(b"#!/host/path/python\nprint('ready')\n")
    script.chmod(0o755)
    metadata = site / "sample_tool-1.0.dist-info"
    metadata.mkdir()
    (metadata / "entry_points.txt").write_text("[console_scripts]\nsample-tool = sample_tool:main\n")
    _stage_console_scripts(site, staging, "/work/.venv")
    staged = staging / "work/.venv/bin/sample-tool"
    assert staged.read_bytes() == b"#!/work/.venv/bin/python\nprint('ready')\n"
    assert staged.stat().st_mode & 0o111
    assert not scripts.exists()


@pytest.mark.parametrize(
    ("name", "shebang"),
    [("activate", b"#!/host/path/python\n"), ("sample-tool", b"#!/bin/sh\n")],
)
def test_console_script_relocation_rejects_reserved_or_non_python_scripts(
    tmp_path: Path, name: str, shebang: bytes
) -> None:
    staging = tmp_path / "staging"
    site = staging / "work/.venv/lib/python3.13/site-packages"
    scripts = site / "bin"
    scripts.mkdir(parents=True)
    script = scripts / name
    script.write_bytes(shebang + b"echo ready\n")
    metadata = site / "sample_tool-1.0.dist-info"
    metadata.mkdir()
    (metadata / "entry_points.txt").write_text(f"[console_scripts]\n{name} = sample_tool:main\n")
    with pytest.raises(shellsim.PackageInstallError):
        _stage_console_scripts(site, staging, "/work/.venv")
    assert script.read_bytes() == shebang + b"echo ready\n"


def test_universe_rejects_mismatched_abi_and_path_traversal(tmp_path: Path) -> None:
    root = make_catalog(tmp_path / "wrong-abi", abi="different")
    with pytest.raises(shellsim.PackageInstallError, match="ABI"):
        Universe(root, abi=ABI, python_version="3.13.7")

    root = make_catalog(
        tmp_path / "traversal",
        packages=[{"name": "native", "version": "1.0", "wheel": "../outside.whl", "sha256": "0" * 64}],
    )
    with pytest.raises(shellsim.PackageInstallError, match="traversal"):
        Universe(root, abi=ABI, python_version="3.13.7")


def test_selected_wheel_rejects_corrupt_hash(tmp_path: Path) -> None:
    root = make_catalog(
        tmp_path / "corrupt",
        packages=[{"name": "native", "version": "1.0", "wheel": "wheels/native-1.0.whl", "sha256": "0" * 64}],
    )
    wheel = root / "wheels/native-1.0.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"bad wheel")
    universe = Universe(root, abi=ABI, python_version="3.13.7")
    selected = universe.packages[("native", "1.0")]
    with pytest.raises(shellsim.PackageInstallError, match="SHA-256 mismatch"):
        _verify_file(selected["path"], selected["sha256"], 1024)


def test_universe_rejects_missing_native_closure(tmp_path: Path) -> None:
    root = make_catalog(tmp_path / "closure")
    universe = Universe(root, abi=ABI, python_version="3.13.7")
    with pytest.raises(shellsim.PackageInstallError, match="missing native provider"):
        universe.provider_closure({"libmissing.so"})


def test_native_wasm_must_match_core_abi() -> None:
    _verify_wasm(marked_wasm(), ABI)
    with pytest.raises(shellsim.PackageInstallError, match="ABI"):
        _verify_wasm(marked_wasm("other-abi"), ABI)
    with pytest.raises(shellsim.PackageInstallError, match="core Wasm"):
        _verify_wasm(b"host ELF library", ABI)


def test_native_wheel_rejects_undeclared_extension(tmp_path: Path) -> None:
    path = tmp_path / "native-1.0-cp313-cp313-wasm32_wasip1.whl"
    info = "native-1.0.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("native.so", marked_wasm())
        archive.writestr(f"{info}/METADATA", "Metadata-Version: 2.3\nName: native\nVersion: 1.0\n")
        archive.writestr(f"{info}/WHEEL", "Root-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n")
        archive.writestr(
            f"{info}/shellsim-native.json",
            json.dumps(
                {"schema_version": 1, "name": "native", "version": "1.0", "abi": ABI, "recipe": {}, "artifacts": []}
            ),
        )
    with pytest.raises(shellsim.PackageInstallError, match="undeclared native files"):
        _inspect_wheel(path, name="native", version="1.0", abi=ABI, curated=True)


def test_wheel_tag_must_be_an_exact_metadata_line(tmp_path: Path) -> None:
    path = tmp_path / "native-1.0-cp313-cp313-wasm32_wasip1.whl"
    info = "native-1.0.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{info}/METADATA", "Metadata-Version: 2.3\nName: native\nVersion: 1.0\n")
        archive.writestr(f"{info}/WHEEL", "Root-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1_unsafe\n")
    with pytest.raises(shellsim.PackageInstallError, match="incompatible WASI tag"):
        _inspect_wheel(path, name="native", version="1.0", abi=ABI, curated=True)


def test_wheel_set_limits_and_member_conflicts_are_checked_before_unpacking() -> None:
    members: dict[str, tuple[bool, str | None]] = {}
    first = _WheelInspection(set(), {}, {"shared/__init__.py": (False, "a" * 64)}, 1, 70 * 1024 * 1024)
    count, size = _add_wheel_to_set(first, members, 0, 0)
    identical = _WheelInspection(set(), {}, {"shared/__init__.py": (False, "a" * 64)}, 1, 1)
    assert _add_wheel_to_set(identical, members, count, size) == (2, size + 1)
    changed = _WheelInspection(set(), {}, {"shared/__init__.py": (False, "b" * 64)}, 1, 1)
    with pytest.raises(shellsim.PackageInstallError, match="path conflicts"):
        _add_wheel_to_set(changed, members, count, size)
    extra = _WheelInspection(set(), {}, {"other.py": (False, "c" * 64)}, 1, 70 * 1024 * 1024)
    with pytest.raises(shellsim.PackageInstallError, match="uncompressed size limit"):
        _add_wheel_to_set(extra, members, count, size)
    too_many = _WheelInspection(set(), {}, {}, 10_000, 0)
    with pytest.raises(shellsim.PackageInstallError, match="file or uncompressed size limit"):
        _add_wheel_to_set(too_many, members, count, size)


def test_provider_changed_after_closure_is_rejected_at_staging(tmp_path: Path) -> None:
    provider_data = marked_wasm()
    root = make_catalog(
        tmp_path / "provider",
        native_providers=[
            {
                "name": "libexample.so",
                "path": "providers/libexample.so",
                "destination": "/lib/libexample.so",
                "sha256": hashlib.sha256(provider_data).hexdigest(),
                "native_dependencies": [],
            }
        ],
    )
    source = root / "providers/libexample.so"
    source.parent.mkdir()
    source.write_bytes(provider_data)
    universe = Universe(root, abi=ABI, python_version="3.13.7")
    closure = universe.provider_closure({"libexample.so"})
    source.write_bytes(marked_wasm("other-abi"))
    destination = tmp_path / "staging/lib/libexample.so"
    destination.parent.mkdir(parents=True)
    with pytest.raises(shellsim.PackageInstallError, match="SHA-256 mismatch"):
        _stage_provider(universe, "libexample.so", closure["libexample.so"], destination)
    assert not destination.exists()


@pytest.fixture
def live_universe() -> tuple[shellsim.CPythonRuntime, shellsim.Environment]:
    bundle = os.environ.get("SHELLSIM_DYNAMIC_V2_BUNDLE")
    catalog = os.environ.get("SHELLSIM_CPYTHON_UNIVERSE")
    uv = os.environ.get("SHELLSIM_PATCHED_UV")
    if not all((bundle, catalog, uv)):
        pytest.skip("set SHELLSIM_DYNAMIC_V2_BUNDLE, SHELLSIM_CPYTHON_UNIVERSE, and SHELLSIM_PATCHED_UV")
    runtime = shellsim.CPythonRuntime(bundle, universe=catalog, uv=uv)
    environment = shellsim.Environment(cpu=4_000_000_000, memory=512 * 1024 * 1024, disk=128 * 1024 * 1024)
    runtime.mount(environment)
    return runtime, environment


def test_zlib_wrapper_installs_separate_native_provider(live_universe) -> None:
    runtime, environment = live_universe
    runtime.install_pypi(environment, "wasm-zlib-wrapper==0.1")
    assert environment.read_file("/lib/libz.so").startswith(b"\0asm\x01\0\0\0")
    result = runtime.run(
        environment,
        ["-c", "import wasm_zlib_wrapper; assert wasm_zlib_wrapper.roundtrip(b'abc' * 100) == b'abc' * 100"],
    )
    assert result.returncode == 0, result.stderr


def test_magiccube_resolves_transitive_native_numpy(live_universe) -> None:
    runtime, environment = live_universe
    interpreter = Path(runtime.bundle / "rootfs/usr/bin/python3.wasm")
    before = hashlib.sha256(interpreter.read_bytes()).hexdigest()
    runtime.install_pypi(environment, "magiccube==0.3.0")
    result = runtime.run(
        environment,
        [
            "-c",
            "import magiccube, numpy as np, numpy._core._multiarray_umath as core; "
            "a=np.arange(12).reshape(3,4); assert a.sum(axis=0).tolist()==[12,15,18,21]; "
            "c=magiccube.Cube(3); assert c.is_done(); c.rotate('R U F'); assert not c.is_done(); "
            "c.rotate(\"F' U' R'\"); assert c.is_done(); assert core.__spec__.origin.endswith('.so')",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256(interpreter.read_bytes()).hexdigest() == before


def test_environment_uses_mounted_cpython_for_python_and_console_commands(live_universe) -> None:
    runtime, environment = live_universe
    environment.install_pypi("pytest==8.4.1")
    result = environment.run_python(
        "import sys; print(sys.prefix); print(sys.argv); print(sys.stdin.read())",
        argv=("argument with spaces",),
        stdin=b"guest input",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"/work/.venv\n['-c', 'argument with spaces']\nguest input\n"
    assert environment.run("pytest --version").stdout == b"pytest 8.4.1\n"
    child = environment.run(
        'python -c \'import subprocess; r=subprocess.run(["pytest", "--version"], '
        "capture_output=True, text=True); print(r.returncode); print(r.stdout.strip())'"
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout == b"0\npytest 8.4.1\n"


def test_unavailable_curated_numpy_version_does_not_mutate_vfs(live_universe) -> None:
    runtime, environment = live_universe
    with pytest.raises(shellsim.PackageInstallError, match="No solution found"):
        runtime.install_pypi(environment, "numpy==2.2.0")
    with pytest.raises(shellsim.SimulationError):
        environment.read_file(runtime.site_packages + "/numpy/__init__.py")
    result = runtime.run(environment, ["-c", "import sys; print(sys.platform)"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"wasi\n"


def test_conflict_preserves_existing_files_and_skips_extension(live_universe) -> None:
    runtime, environment = live_universe
    assert runtime.universe is not None
    catalog = Universe(runtime.universe, abi=runtime.manifest["dynamic_abi"], python_version=runtime.version)
    existing_provider = catalog.providers["libz.so"]["path"].read_bytes()
    environment.mkdir("/lib", parents=True)
    environment.write_file("/lib/libz.so", existing_provider)
    conflict = runtime.site_packages + "/wasm_zlib_wrapper/__init__.py"
    environment.mkdir(runtime.site_packages + "/wasm_zlib_wrapper", parents=True)
    environment.write_file(conflict, b"user data")
    with pytest.raises(shellsim.PackageInstallError, match="overwrite"):
        runtime.install_pypi(environment, "wasm-zlib-wrapper==0.1")
    assert environment.read_file(conflict) == b"user data"
    assert environment.read_file("/lib/libz.so") == existing_provider
    with pytest.raises(shellsim.SimulationError):
        environment.read_file(runtime.site_packages + "/zlib_consumer.so")
