"""Exercise release admission, cache integrity and atomic publication on the host."""

from __future__ import annotations

import hashlib
import json
import stat
import urllib.request
import zipfile
from pathlib import Path

import pytest
import shellsim
from shellsim._cpython_release import _fetch, _ReleaseRedirect
from shellsim.native_packages import _digest

from ports.python.cpython import release

ABI = "shellsim-wasi-sdk34-cpython3137-v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    runtime = tmp_path / "runtime"
    binary = runtime / "rootfs/usr/bin/python3.wasm"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\0asm\x01\0\0\0")
    binary.chmod(0o755)
    (runtime / "manifest.json").write_text(
        json.dumps(
            {
                "recipe": {"version": "3.13.7", "target": "wasm32-wasip1", "prefix": "/usr"},
                "site_packages": "/usr/lib/python3.13/site-packages",
                "dynamic_abi": ABI,
                "files": {"/usr/bin/python3.wasm": _sha(binary)},
            }
        )
    )
    universe = tmp_path / "universe"
    universe.mkdir()
    (universe / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "abi": ABI,
                "target": "wasm32-wasip1",
                "python_version": "3.13.7",
                "packages": [],
                "native_providers": [],
            }
        )
    )
    uv = tmp_path / "uv"
    uv.write_bytes(b"#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    monkeypatch.setattr(release, "_host_requirements", lambda _: ("2.17", ["libc.so.6"]))
    monkeypatch.setattr("shellsim._cpython_release.platform.system", lambda: "Linux")
    monkeypatch.setattr("shellsim._cpython_release.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("shellsim._cpython_release.platform.libc_ver", lambda: ("glibc", "2.43"))
    return runtime, universe, uv


def _built(release_inputs: tuple[Path, Path, Path], destination: Path) -> Path:
    return release.build_release(*release_inputs, destination)


def _marked_wasm() -> bytes:
    marker = b"\x0cshellsim.abi" + ABI.encode()
    return b"\0asm\x01\0\0\0" + bytes((0, len(marker))) + marker


def _provider(universe: Path, name: str, dependencies: list[str]) -> dict:
    path = universe / name
    path.write_bytes(_marked_wasm())
    return {
        "name": name,
        "path": name,
        "destination": "/lib/" + name,
        "sha256": _sha(path),
        "native_dependencies": dependencies,
    }


def _native_catalog(root: Path, *, variants: list[tuple[str, bool]] | None = None) -> Path:
    root.mkdir()
    records = []
    for version, unresolved in variants or [("4.4.1", False)]:
        artifact = root / "artifacts" / version
        tool = artifact / "bin/make"
        tool.parent.mkdir(parents=True)
        tool.write_bytes(("guest make " + version).encode())
        recipe = {
            "name": "make",
            "version": version,
            "target": "wasm32-wasip1",
            "target_profile": "fixture-v1",
            "exports": {"tools": ["bin/make"]},
            "target_dependencies": [],
        }
        inputs = {
            "recipe": recipe,
            "recipe_sha256": _digest(recipe),
            "toolchain": {"profile": {"name": "fixture-v1", "target": "wasm32-wasip1"}},
            "dependency_artifacts": {},
        }
        manifest = {"inputs": inputs, "files": {"bin/make": _sha(tool)}}
        manifest["artifact_sha256"] = _digest(manifest)
        (artifact / "artifact.json").write_text(json.dumps(manifest))
        records.append(
            {
                "name": "make",
                "version": version,
                "kind": "build-tool",
                "artifact": "artifacts/" + version,
                "artifact_sha256": manifest["artifact_sha256"],
                "destinations": {"bin/make": "/usr/bin/make"},
                "dependencies": ([{"requirement": "missing-provider==1", "kind": "build-tool"}] if unresolved else []),
                "recipe_name": "make",
                "target_profile": "fixture-v1",
                "toolchain_sha256": _digest(inputs["toolchain"]),
            }
        )
    (root / "catalog.json").write_text(json.dumps({"format": 1, "target": "wasm32-wasip1", "packages": records}))
    return root / "catalog.json"


def _edit_archive(descriptor: Path, edit) -> None:
    archive = descriptor.parent / "cohort.zip"
    with zipfile.ZipFile(archive) as source:
        members = [(item, source.read(item)) for item in source.infolist()]
    with zipfile.ZipFile(archive, "w") as output:
        for item, data in members:
            output.writestr(item, data)
        edit(output)
    data = json.loads(descriptor.read_text())
    data["archive"].update({"size": archive.stat().st_size, "sha256": _sha(archive)})
    descriptor.write_text(json.dumps(data))


def test_release_is_deterministic_and_offline_cache_is_verified(release_inputs, tmp_path):
    first = _built(release_inputs, tmp_path / "first")
    second = _built(release_inputs, tmp_path / "second")
    assert _sha(first.parent / "cohort.zip") == _sha(second.parent / "cohort.zip")
    assert first.read_bytes() == second.read_bytes()
    cache = tmp_path / "cache"
    runtime = shellsim.CPythonRuntime.from_release(first, cache_dir=cache)
    assert runtime.manifest["dynamic_abi"] == ABI
    (first.parent / "cohort.zip").unlink()
    (first.parent / "uv-linux-x86_64-glibc").unlink()
    assert shellsim.CPythonRuntime.from_release(first, cache_dir=cache, offline=True).bundle == runtime.bundle
    (runtime.bundle / "rootfs/usr/bin/python3.wasm").write_bytes(b"changed")
    with pytest.raises(shellsim.PackageInstallError, match="cached CPython release is corrupt"):
        shellsim.CPythonRuntime.from_release(first, cache_dir=cache, offline=True)


def test_threaded_release_keeps_wheel_platform_and_pins_runtime_target(release_inputs, tmp_path):
    runtime, universe, _ = release_inputs
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["recipe"]["target"] = "wasm32-wasip1-threads"
    manifest["recipe"]["dynamic_abi"] = "shellsim-wasi-sdk34-cpython3137-threads-v3"
    manifest["dynamic_abi"] = manifest["recipe"]["dynamic_abi"]
    manifest_path.write_text(json.dumps(manifest))
    catalog_path = universe / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["abi"] = manifest["dynamic_abi"]
    catalog_path.write_text(json.dumps(catalog))

    descriptor = _built(release_inputs, tmp_path / "release")
    data = json.loads(descriptor.read_text())
    assert data["target"] == "wasm32-wasip1"
    assert data["runtime_target"] == "wasm32-wasip1-threads"
    loaded = shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache")
    assert loaded.manifest["dynamic_abi"] == manifest["dynamic_abi"]

    data["runtime_target"] = "wasm32-wasip1"
    descriptor.write_text(json.dumps(data))
    with pytest.raises(shellsim.PackageInstallError):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "other-cache")


def test_base_release_descriptor_without_runtime_target_still_loads(release_inputs, tmp_path):
    descriptor = _built(release_inputs, tmp_path / "release")
    data = json.loads(descriptor.read_text())
    del data["runtime_target"]
    descriptor.write_text(json.dumps(data))
    runtime = shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache")
    assert runtime.manifest["recipe"]["target"] == "wasm32-wasip1"


def test_release_rejects_threaded_runtime_with_base_catalog(release_inputs, tmp_path):
    runtime, _, _ = release_inputs
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["recipe"]["target"] = "wasm32-wasip1-threads"
    manifest["dynamic_abi"] = "shellsim-wasi-sdk34-cpython3137-threads-v3"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "release"
    with pytest.raises(shellsim.PackageInstallError):
        _built(release_inputs, output)
    assert not output.exists()


def test_cached_runtime_cannot_be_changed_with_its_manifest(release_inputs, tmp_path):
    descriptor = _built(release_inputs, tmp_path / "release")
    runtime = shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache")
    binary = runtime.bundle / "rootfs/usr/bin/python3.wasm"
    binary.write_bytes(b"\0asm\x01\0\0\0changed")
    manifest = json.loads((runtime.bundle / "manifest.json").read_text())
    manifest["files"]["/usr/bin/python3.wasm"] = _sha(binary)
    (runtime.bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(shellsim.PackageInstallError, match="cached CPython release is corrupt"):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache", offline=True)


def test_cached_wheel_cannot_be_changed_with_its_catalog(release_inputs, tmp_path):
    runtime, universe, uv = release_inputs
    wheel = universe / "sample-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sample-1.0.dist-info/METADATA", "Name: sample\nVersion: 1.0\n")
        archive.writestr("sample-1.0.dist-info/WHEEL", "Root-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("sample/__init__.py", "VALUE = 1\n")
    catalog = json.loads((universe / "catalog.json").read_text())
    catalog["packages"] = [{"name": "sample", "version": "1.0", "wheel": wheel.name, "sha256": _sha(wheel)}]
    (universe / "catalog.json").write_text(json.dumps(catalog))
    descriptor = _built((runtime, universe, uv), tmp_path / "release")
    cached = shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache")
    staged_wheel = cached.universe / wheel.name
    staged_wheel.write_bytes(b"changed wheel")
    staged_catalog = cached.universe / "catalog.json"
    catalog = json.loads(staged_catalog.read_text())
    catalog["packages"][0]["sha256"] = _sha(staged_wheel)
    staged_catalog.write_text(json.dumps(catalog))
    with pytest.raises(shellsim.PackageInstallError, match="cached CPython release is corrupt"):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache", offline=True)


def test_asset_hash_failure_does_not_publish_cache(release_inputs, tmp_path):
    descriptor = _built(release_inputs, tmp_path / "release")
    data = json.loads(descriptor.read_text())
    data["archive"]["sha256"] = "0" * 64
    descriptor.write_text(json.dumps(data))
    cache = tmp_path / "cache"
    with pytest.raises(shellsim.PackageInstallError, match="SHA-256 mismatch"):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=cache)
    assert not list(cache.iterdir())


@pytest.mark.parametrize("member", ["../escape", "runtime/rootfs/../escape", "runtime/symlink"])
def test_archive_rejects_unsafe_extra_members(release_inputs, tmp_path, member):
    descriptor = _built(release_inputs, tmp_path / "release")

    def edit(archive):
        info = zipfile.ZipInfo(member)
        info.external_attr = ((stat.S_IFLNK | 0o777) if member.endswith("symlink") else stat.S_IFREG | 0o644) << 16
        archive.writestr(info, b"unsafe")

    _edit_archive(descriptor, edit)
    with pytest.raises(shellsim.PackageInstallError, match="archive contains"):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache")


def test_release_rejects_missing_catalogued_wheel(release_inputs, tmp_path):
    descriptor = _built(release_inputs, tmp_path / "release")
    archive = descriptor.parent / "cohort.zip"
    with zipfile.ZipFile(archive) as source:
        members = [(item, source.read(item)) for item in source.infolist()]
    files = {item.filename: content for item, content in members}
    catalog = json.loads(files["universe/catalog.json"])
    catalog["packages"] = [
        {"name": "missing", "version": "1.0", "wheel": "missing-1.0-py3-none-any.whl", "sha256": "0" * 64}
    ]
    files["universe/catalog.json"] = json.dumps(catalog).encode()
    with zipfile.ZipFile(archive, "w") as output:
        for item, _ in members:
            output.writestr(item, files[item.filename])
    data = json.loads(descriptor.read_text())
    data["archive"].update({"size": archive.stat().st_size, "sha256": _sha(archive)})
    data["catalog_sha256"] = hashlib.sha256(files["universe/catalog.json"]).hexdigest()
    descriptor.write_text(json.dumps(data))
    with pytest.raises(shellsim.PackageInstallError, match="curated artifact"):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache")


def test_release_rejects_native_wheel_with_missing_provider(release_inputs, tmp_path):
    runtime, universe, uv = release_inputs
    wheel = universe / "probe-1.0-cp313-cp313-wasm32_wasip1.whl"
    extension = _marked_wasm()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("probe/extension.so", extension)
        archive.writestr("probe-1.0.dist-info/METADATA", "Name: probe\nVersion: 1.0\n")
        archive.writestr(
            "probe-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
        )
        archive.writestr(
            "probe-1.0.dist-info/shellsim-native.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "probe",
                    "version": "1.0",
                    "abi": ABI,
                    "recipe": {},
                    "artifacts": [
                        {
                            "path": "probe/extension.so",
                            "sha256": hashlib.sha256(extension).hexdigest(),
                            "native_dependencies": ["libmissing.so"],
                        }
                    ],
                }
            ),
        )
        archive.writestr("probe-1.0.dist-info/RECORD", "")
    catalog = json.loads((universe / "catalog.json").read_text())
    catalog["packages"] = [{"name": "probe", "version": "1.0", "wheel": wheel.name, "sha256": _sha(wheel)}]
    (universe / "catalog.json").write_text(json.dumps(catalog))
    output = tmp_path / "release"
    with pytest.raises(shellsim.PackageInstallError, match="missing native provider"):
        release.build_release(runtime, universe, uv, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("providers", "error"),
    [
        ([("liborphan.so", ["libmissing.so"])], "missing native provider"),
        ([("liba.so", ["libb.so"]), ("libb.so", ["liba.so"])], "native provider dependency cycle"),
    ],
)
def test_release_rejects_broken_unselected_provider_closure(release_inputs, tmp_path, providers, error):
    runtime, universe, uv = release_inputs
    catalog = json.loads((universe / "catalog.json").read_text())
    catalog["native_providers"] = [_provider(universe, name, dependencies) for name, dependencies in providers]
    (universe / "catalog.json").write_text(json.dumps(catalog))
    output = tmp_path / "release"
    with pytest.raises(shellsim.PackageInstallError, match=error):
        release.build_release(runtime, universe, uv, output)
    assert not output.exists()


def test_native_release_loads_without_python_resolver_and_installs_tool(release_inputs, tmp_path, monkeypatch):
    native = _native_catalog(tmp_path / "native")
    descriptor = release.build_release(*release_inputs, tmp_path / "release", native_catalog=native)
    data = json.loads(descriptor.read_text())
    descriptor.write_text(json.dumps({key: data[key] for key in ("schema_version", "target", "native")}))
    monkeypatch.setattr("shellsim._cpython_release.platform.system", lambda: "Darwin")
    cache = tmp_path / "cache"
    with pytest.raises(shellsim.PackageInstallError, match="not cached"):
        shellsim.NativePackageUniverse.from_release(descriptor, cache_dir=cache, offline=True)
    universe = shellsim.NativePackageUniverse.from_release(descriptor, cache_dir=cache)
    environment = shellsim.Environment()
    assert universe.install(environment, "make>=4.4,<5") == {"make": "4.4.1"}
    assert environment.read_file("/usr/bin/make") == b"guest make 4.4.1"
    (descriptor.parent / "native.zip").unlink()
    assert shellsim.NativePackageUniverse.from_release(descriptor, cache_dir=cache, offline=True).catalog_path.is_file()
    (universe.root / "artifacts/4.4.1/bin/make").write_bytes(b"changed")
    with pytest.raises(shellsim.PackageInstallError, match="cached native release is corrupt"):
        shellsim.NativePackageUniverse.from_release(descriptor, cache_dir=cache, offline=True)


def test_python_only_release_has_no_native_catalog(release_inputs, tmp_path):
    descriptor = _built(release_inputs, tmp_path / "release")
    with pytest.raises(shellsim.PackageInstallError, match="no native package catalog"):
        shellsim.NativePackageUniverse.from_release(descriptor, cache_dir=tmp_path / "cache")


def test_cached_native_artifact_cannot_be_resealed_without_descriptor(release_inputs, tmp_path):
    native = _native_catalog(tmp_path / "native")
    descriptor = release.build_release(*release_inputs, tmp_path / "release", native_catalog=native)
    universe = shellsim.NativePackageUniverse.from_release(descriptor, cache_dir=tmp_path / "cache")
    tool = universe.root / "artifacts/4.4.1/bin/make"
    tool.write_bytes(b"changed")
    manifest_path = universe.root / "artifacts/4.4.1/artifact.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["bin/make"] = _sha(tool)
    manifest.pop("artifact_sha256")
    manifest["artifact_sha256"] = _digest(manifest)
    manifest_path.write_text(json.dumps(manifest))
    catalog = json.loads(universe.catalog_path.read_text())
    catalog["packages"][0]["artifact_sha256"] = manifest["artifact_sha256"]
    universe.catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(shellsim.PackageInstallError, match="cached native release is corrupt"):
        shellsim.NativePackageUniverse.from_release(descriptor, cache_dir=tmp_path / "cache", offline=True)


def test_native_release_rejects_unsatisfied_candidate_before_publication(release_inputs, tmp_path):
    native = _native_catalog(tmp_path / "native", variants=[("4.4.1", False), ("4.4.2", True)])
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="no compatible native package closure"):
        release.build_release(*release_inputs, output, native_catalog=native)
    assert not output.exists()


def test_native_release_checks_base_version_despite_valid_local_version(release_inputs, tmp_path):
    native = _native_catalog(tmp_path / "native", variants=[("1.0", True), ("1.0+local", False)])
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="no compatible native package closure"):
        release.build_release(*release_inputs, output, native_catalog=native)
    assert not output.exists()


def test_release_preserves_broken_output_symlink(release_inputs, tmp_path):
    output = tmp_path / "release"
    missing = tmp_path / "missing"
    output.symlink_to(missing, target_is_directory=True)
    with pytest.raises(ValueError, match="output already exists"):
        release.build_release(*release_inputs, output)
    assert output.is_symlink()
    assert output.resolve() == missing


def test_native_release_archive_cannot_add_python_files(release_inputs, tmp_path):
    native = _native_catalog(tmp_path / "native")
    descriptor = release.build_release(*release_inputs, tmp_path / "release", native_catalog=native)
    archive = descriptor.parent / "native.zip"
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("runtime/host-escape", b"unexpected")
    data = json.loads(descriptor.read_text())
    data["native"]["archive"].update({"sha256": _sha(archive), "size": archive.stat().st_size})
    descriptor.write_text(json.dumps(data))
    cache = tmp_path / "cache"
    with pytest.raises(shellsim.PackageInstallError, match="unsafe path"):
        shellsim.NativePackageUniverse.from_release(descriptor, cache_dir=cache)
    assert not list(cache.iterdir())


def test_unsupported_host_and_offline_miss_do_not_fetch(release_inputs, tmp_path, monkeypatch):
    descriptor = _built(release_inputs, tmp_path / "release")
    monkeypatch.setattr("shellsim._cpython_release.platform.system", lambda: "Darwin")
    with pytest.raises(shellsim.PackageInstallError, match="no patched WASI resolver"):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache")
    monkeypatch.setattr("shellsim._cpython_release.platform.system", lambda: "Linux")
    with pytest.raises(shellsim.PackageInstallError, match="not cached"):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=tmp_path / "cache", offline=True)


def test_cache_entry_symlink_is_rejected(release_inputs, tmp_path):
    descriptor = _built(release_inputs, tmp_path / "release")
    cache = tmp_path / "cache"
    runtime = shellsim.CPythonRuntime.from_release(descriptor, cache_dir=cache)
    entry = runtime.bundle.parent
    shadow = tmp_path / "shadow"
    entry.rename(shadow)
    entry.symlink_to(shadow, target_is_directory=True)
    with pytest.raises(shellsim.PackageInstallError, match="cached CPython release is corrupt"):
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=cache, offline=True)


def test_relative_cache_directory_supports_catalogued_wheels(release_inputs, tmp_path, monkeypatch):
    runtime, universe, uv = release_inputs
    wheel = universe / "sample-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sample-1.0.dist-info/METADATA", "Name: sample\nVersion: 1.0\n")
        archive.writestr("sample-1.0.dist-info/WHEEL", "Root-Is-Purelib: true\nTag: py3-none-any\n")
    catalog = json.loads((universe / "catalog.json").read_text())
    catalog["packages"] = [{"name": "sample", "version": "1.0", "wheel": wheel.name, "sha256": _sha(wheel)}]
    (universe / "catalog.json").write_text(json.dumps(catalog))
    descriptor = _built((runtime, universe, uv), tmp_path / "release")
    monkeypatch.chdir(tmp_path)
    first = shellsim.CPythonRuntime.from_release(descriptor, cache_dir=Path("cache"))
    assert (
        shellsim.CPythonRuntime.from_release(descriptor, cache_dir=Path("cache"), offline=True).bundle == first.bundle
    )


def test_release_fetch_enforces_declared_bytes_and_https_redirects(release_inputs, tmp_path):
    descriptor = _built(release_inputs, tmp_path / "release")
    asset = {"url": "cohort.zip", "size": 1, "sha256": "0" * 64}
    with pytest.raises(shellsim.PackageInstallError, match="declared size"):
        _fetch(asset, descriptor, tmp_path / "download")
    request = urllib.request.Request("https://github.com/org/repo/releases/download/v1/cohort.zip")
    with pytest.raises(shellsim.PackageInstallError, match="redirected"):
        _ReleaseRedirect().redirect_request(request, None, 302, "redirect", {}, "http://127.0.0.1/private")
