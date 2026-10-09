"""Curated provenance applies independently to pure and native wheel kinds."""

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest
import shellsim
from shellsim._cpython_universe import Universe, _inspect_wheel, _verify_file

ABI = "shellsim-wasi-sdk34-cpython3137-v2"


def pure_wheel(root, *, hidden=False):
    path = root / "example-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("example.py", "VALUE = 1\n")
        archive.writestr("example-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: example\nVersion: 1.0\n")
        archive.writestr("example-1.0.dist-info/WHEEL", "Root-Is-Purelib: true\nTag: py3-none-any\n")
        if hidden:
            archive.writestr("example/hidden.so", b"\0asm\x01\0\0\0")
    return path


def test_curated_pure_wheel_preserves_standard_tag(tmp_path):
    path = pure_wheel(tmp_path)
    result = _inspect_wheel(path, name="example", version="1.0", abi=ABI, curated=True)
    assert result.artifacts == {}
    assert result.dependencies == set()
    assert result.members["example.py"][0] is False


@pytest.mark.parametrize("curated", [True, False])
def test_hidden_native_file_rejected_in_pure_wheel(tmp_path, curated):
    path = pure_wheel(tmp_path, hidden=True)
    with pytest.raises(shellsim.PackageInstallError, match="pure wheel contains native files"):
        _inspect_wheel(path, name="example", version="1.0", abi=ABI, curated=curated)


def test_curated_pure_identity_remains_authoritative(tmp_path):
    path = pure_wheel(tmp_path)
    catalog = {
        "schema_version": 1,
        "abi": ABI,
        "target": "wasm32-wasip1",
        "python_version": "3.13.7",
        "packages": [{"name": "example", "version": "1.0", "wheel": path.name, "sha256": "0" * 64}],
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    selected = Universe(tmp_path, abi=ABI, python_version="3.13.7").packages[("example", "1.0")]
    with pytest.raises(shellsim.PackageInstallError, match="SHA-256 mismatch"):
        _verify_file(selected["path"], selected["sha256"], 1024 * 1024)
    with pytest.raises(shellsim.PackageInstallError, match="identity differs"):
        _inspect_wheel(path, name="other", version="1.0", abi=ABI, curated=True)


def test_real_curated_pytest_with_uncurated_pure_dependencies():
    keys = ("SHELLSIM_DYNAMIC_V2_BUNDLE", "SHELLSIM_CURATED_PURE_UNIVERSE", "SHELLSIM_PATCHED_UV")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("set real bundle, curated pytest universe and patched uv paths")
    root = Path(os.environ[keys[1]])
    catalog = json.loads((root / "catalog.json").read_text())
    pytest_entry = next(entry for entry in catalog["packages"] if entry["name"] == "pytest")
    assert pytest_entry["version"] == "8.4.1"
    runtime = shellsim.CPythonRuntime(os.environ[keys[0]], universe=root, uv=os.environ[keys[2]])
    image = runtime.bundle / "rootfs/usr/bin/python3.wasm"
    image_before = hashlib.sha256(image.read_bytes()).hexdigest()
    env = shellsim.Environment(cpu=10_000_000_000, memory=1024**3, disk=256 * 1024**2)
    runtime.mount(env)
    runtime.install_pypi(env, "pytest==8.4.1")
    dependencies = {"pluggy", "iniconfig", "packaging", "pygments"}
    assert dependencies.isdisjoint({entry["name"] for entry in catalog["packages"]})
    env.write_file("/work/test_example.py", b"def test_value():\n    assert 6 * 7 == 42\n")
    result = runtime.run(env, ["-m", "pytest", "-q", "/work/test_example.py"])
    assert result.returncode == 0, result.stderr
    assert b"1 passed" in result.stdout
    env.write_file("/work/test_example.py", b"def test_value():\n    assert 6 * 7 == 43\n")
    failure = runtime.run(env, ["-m", "pytest", "-q", "/work/test_example.py"])
    assert failure.returncode == 1, failure.stderr
    assert b"1 failed" in failure.stdout
    assert hashlib.sha256(image.read_bytes()).hexdigest() == image_before


@pytest.mark.parametrize("failure", ["hash", "unavailable-version"])
def test_public_curated_pure_rejection_preserves_guest(tmp_path, failure):
    keys = ("SHELLSIM_DYNAMIC_V2_BUNDLE", "SHELLSIM_CURATED_PURE_UNIVERSE", "SHELLSIM_PATCHED_UV")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("set real bundle, curated pytest universe and patched uv paths")
    original = Path(os.environ[keys[1]])
    fallback = original / "pure-wheels/pytest-9.1.1-py3-none-any.whl"
    if failure == "unavailable-version" and not fallback.is_file():
        pytest.skip("ordinary pure index must include the real pytest9.1.1 fallback")
    root = tmp_path / "universe"
    shutil.copytree(original, root)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entry = next(item for item in catalog["packages"] if item["name"] == "pytest")
    assert entry["version"] == "8.4.1"
    if failure == "hash":
        entry["sha256"] = "0" * 64
        catalog_path.write_text(json.dumps(catalog))
        request = "pytest==8.4.1"
    else:
        # This version is genuinely present in the separate dependency index,
        # but the authoritative curated project only supplies 8.4.1.
        assert "pytest-9.1.1" in (root / "pure-simple/pytest/index.html").read_text()
        request = "pytest==9.1.1"
    runtime = shellsim.CPythonRuntime(os.environ[keys[0]], universe=root, uv=os.environ[keys[2]])
    env = shellsim.Environment(cpu=10_000_000_000, memory=1024**3, disk=256 * 1024**2)
    runtime.mount(env)
    env.write_file("/sentinel", b"unchanged")
    with pytest.raises(shellsim.PackageInstallError):
        runtime.install_pypi(env, request)
    assert env.read_file("/sentinel") == b"unchanged"
    for name in ("pytest", "_pytest", "pluggy", "iniconfig"):
        assert env.run("test ! -e " + runtime.site_packages + "/" + name).returncode == 0
