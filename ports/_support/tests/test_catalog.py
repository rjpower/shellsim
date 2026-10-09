"""Exercise catalog composition at package identity and native closure boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
import shellsim
from shellsim import PackageInstallError
from shellsim._cpython_universe import Universe

from ports._support.catalog import compose

ABI = "shellsim-wasi-sdk34-cpython3137-v2"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle(root: Path, *, provider: bytes | None = None) -> Path:
    binary = root / "rootfs/usr/bin/python3.wasm"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\0asm\x01\0\0\0")
    binary.chmod(0o755)
    files = {"/usr/bin/python3.wasm": digest(binary.read_bytes())}
    if provider is not None:
        native = root / "rootfs/lib/libexample.so"
        native.parent.mkdir(parents=True)
        native.write_bytes(provider)
        files["/lib/libexample.so"] = digest(provider)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "recipe": {"version": "3.13.7", "target": "wasm32-wasip1", "prefix": "/usr"},
                "site_packages": "/usr/lib/python3.13/site-packages",
                "dynamic_abi": ABI,
                "files": files,
            }
        )
    )
    return root


def wasm() -> bytes:
    marker = b"\x0cshellsim.abi" + ABI.encode()
    return b"\0asm\x01\0\0\0" + bytes((0, len(marker))) + marker


def wheel(root: Path, name: str, version: str, *, native_dependency: str | None = None, content: str = "") -> dict:
    filename = (
        f"{name}-{version}-cp313-cp313-wasm32_wasip1.whl" if native_dependency else f"{name}-{version}-py3-none-any.whl"
    )
    path = root / "wheels" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    info = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}/__init__.py", content)
        archive.writestr(f"{info}/METADATA", f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n")
        if native_dependency:
            binary = wasm()
            archive.writestr(f"{name}/extension.so", binary)
            archive.writestr(
                f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
            )
            archive.writestr(
                f"{info}/shellsim-native.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": name,
                        "version": version,
                        "abi": ABI,
                        "recipe": {},
                        "artifacts": [
                            {
                                "path": f"{name}/extension.so",
                                "sha256": digest(binary),
                                "native_dependencies": [native_dependency],
                            }
                        ],
                    }
                ),
            )
        else:
            archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")
    return {"name": name, "version": version, "wheel": "wheels/" + filename, "sha256": digest(path.read_bytes())}


def catalog(root: Path, *, packages: list[dict] | None = None, providers: list[dict] | None = None) -> Path:
    root.mkdir()
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "abi": ABI,
                "target": "wasm32-wasip1",
                "python_version": "3.13.7",
                "packages": packages or [],
                "native_providers": providers or [],
            }
        )
    )
    return root


def update_catalog(root: Path, *, packages: list[dict] | None = None, providers: list[dict] | None = None) -> None:
    data = json.loads((root / "catalog.json").read_text())
    if packages is not None:
        data["packages"] = packages
    if providers is not None:
        data["native_providers"] = providers
    (root / "catalog.json").write_text(json.dumps(data))


def provider(root: Path) -> dict:
    binary = wasm()
    path = root / "providers/libexample.so"
    path.parent.mkdir(parents=True)
    path.write_bytes(binary)
    return {
        "name": "libexample.so",
        "path": "providers/libexample.so",
        "destination": "/lib/libexample.so",
        "sha256": digest(binary),
        "native_dependencies": [],
    }


def test_compose_preserves_declared_native_closure(tmp_path: Path) -> None:
    first = catalog(tmp_path / "first")
    update_catalog(first, packages=[wheel(first, "sample", "1.0")])
    second = catalog(tmp_path / "second")
    second_package = wheel(second, "consumer", "2.0", native_dependency="libexample.so")
    second_provider = provider(second)
    update_catalog(second, packages=[second_package], providers=[second_provider])
    result = compose(bundle(tmp_path / "runtime"), [first, second], tmp_path / "merged")
    universe = Universe(result, abi=ABI, python_version="3.13.7")
    assert universe.provider_closure({"libexample.so"}) == {"libexample.so": result / "providers/libexample.so"}


def test_conflicting_same_version_fails_without_publication(tmp_path: Path) -> None:
    first = catalog(tmp_path / "first")
    second = catalog(tmp_path / "second")
    for root, content in ((first, "first"), (second, "second")):
        package = wheel(root, "sample", "1.0", content=content)
        update_catalog(root, packages=[package])
    output = tmp_path / "merged"
    with pytest.raises(PackageInstallError, match="conflicting curated package version"):
        compose(bundle(tmp_path / "runtime"), [first, second], output)
    assert not output.exists()


def test_equivalent_pep440_versions_cannot_publish_different_bytes(tmp_path: Path) -> None:
    first = catalog(tmp_path / "first")
    second = catalog(tmp_path / "second")
    update_catalog(first, packages=[wheel(first, "sample", "1.0")])
    update_catalog(second, packages=[wheel(second, "sample", "1.0.0")])
    output = tmp_path / "merged"
    with pytest.raises(PackageInstallError, match="conflicting curated package version"):
        compose(bundle(tmp_path / "runtime"), [first, second], output)
    assert not output.exists()


def test_uv_range_selection_installs_each_version_in_a_guest(tmp_path: Path) -> None:
    bundle_path = os.environ.get("SHELLSIM_CATALOG_TEST_BUNDLE")
    uv = os.environ.get("SHELLSIM_PATCHED_UV")
    if not bundle_path or not uv:
        pytest.skip("set SHELLSIM_CATALOG_TEST_BUNDLE and SHELLSIM_PATCHED_UV for the guest resolver test")
    first = catalog(tmp_path / "first")
    second = catalog(tmp_path / "second")
    update_catalog(first, packages=[wheel(first, "catalogselection", "1.0", content="value = 1\n")])
    update_catalog(second, packages=[wheel(second, "catalogselection", "2.0", content="value = 2\n")])
    output = compose(Path(bundle_path), [first, second], tmp_path / "merged")
    runtime = shellsim.CPythonRuntime(bundle_path, universe=output, uv=uv)
    for requirement, expected in (("catalogselection>=1,<2", b"1\n"), ("catalogselection>=2,<3", b"2\n")):
        environment = shellsim.Environment(cpu=4_000_000_000, memory=512 * 1024 * 1024, disk=128 * 1024 * 1024)
        runtime.mount(environment)
        runtime.install_pypi(environment, requirement)
        result = runtime.run(environment, ["-c", "import catalogselection; print(catalogselection.value)"])
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected


def test_missing_native_provider_fails_even_when_runtime_has_library(tmp_path: Path) -> None:
    root = catalog(tmp_path / "consumer")
    package = wheel(root, "consumer", "1.0", native_dependency="libexample.so")
    update_catalog(root, packages=[package])
    output = tmp_path / "merged"
    with pytest.raises(PackageInstallError, match="missing native provider"):
        compose(bundle(tmp_path / "runtime", provider=wasm()), [root], output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("dependency", "error"),
    [("libmissing.so", "missing native provider"), ("libexample.so", "native provider dependency cycle")],
)
def test_unreferenced_provider_dependency_must_close(tmp_path: Path, dependency: str, error: str) -> None:
    root = catalog(tmp_path / "provider")
    entry = provider(root)
    entry["native_dependencies"] = [dependency]
    update_catalog(root, providers=[entry])
    output = tmp_path / "merged"
    with pytest.raises(PackageInstallError, match=error):
        compose(bundle(tmp_path / "runtime"), [root], output)
    assert not output.exists()


def test_runtime_provider_conflict_fails_without_publication(tmp_path: Path) -> None:
    root = catalog(tmp_path / "provider")
    entry = provider(root)
    update_catalog(root, providers=[entry])
    output = tmp_path / "merged"
    with pytest.raises(PackageInstallError, match="conflicts with the runtime"):
        compose(bundle(tmp_path / "runtime", provider=b"different"), [root], output)
    assert not output.exists()


def test_conflicting_provider_bytes_fail_without_publication(tmp_path: Path) -> None:
    first = catalog(tmp_path / "first")
    second = catalog(tmp_path / "second")
    update_catalog(first, providers=[provider(first)])
    changed = provider(second)
    native = second / changed["path"]
    native.write_bytes(native.read_bytes() + b"\x00\x02\x01x")
    changed["sha256"] = digest(native.read_bytes())
    update_catalog(second, providers=[changed])
    output = tmp_path / "merged"
    with pytest.raises(PackageInstallError, match="conflicting native provider"):
        compose(bundle(tmp_path / "runtime"), [first, second], output)
    assert not output.exists()


def test_corrupt_source_fails_without_publication(tmp_path: Path) -> None:
    root = catalog(tmp_path / "source")
    entry = wheel(root, "sample", "1.0")
    update_catalog(root, packages=[entry])
    (root / entry["wheel"]).write_bytes(b"changed")
    output = tmp_path / "merged"
    with pytest.raises(PackageInstallError, match="SHA-256 mismatch"):
        compose(bundle(tmp_path / "runtime"), [root], output)
    assert not output.exists()
