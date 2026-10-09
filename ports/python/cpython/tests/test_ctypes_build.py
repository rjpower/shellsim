"""Reject incompatible native inputs before compiling or publishing a bundle."""

import hashlib
import json
import shutil

import pytest

from ports.native.libffi.shared import build as shared_libffi
from ports.python.cpython import assembly, stdlib_ctypes


@pytest.mark.parametrize("field,value", [("name", "other"), ("target_profile", "host-linux")])
def test_ctypes_rejects_wrong_provider_before_compilation(monkeypatch, tmp_path, field, value):
    recipe = json.loads((shared_libffi.PORT / "recipe.json").read_text())
    recipe[field] = value
    monkeypatch.setattr(stdlib_ctypes, "verify_artifact", lambda path: {"inputs": {"recipe": recipe}})
    monkeypatch.setattr(
        stdlib_ctypes.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("compiler ran for incompatible libffi"),
    )
    with pytest.raises(ValueError, match="pinned libffi provider"):
        stdlib_ctypes.build_ctypes(tmp_path / "base", tmp_path / "runtime", tmp_path / "libffi", tmp_path)
    assert not (tmp_path / "native-artifacts").exists()


def test_shared_libffi_rejects_wrong_static_provider_before_compilation(monkeypatch, tmp_path):
    monkeypatch.setattr(shared_libffi, "verify_artifact", lambda path: {"inputs": {"recipe": {"name": "other"}}})
    monkeypatch.setattr(
        shared_libffi.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("linker ran for incompatible libffi"),
    )
    with pytest.raises(ValueError, match="pinned static provider"):
        shared_libffi.build_shared(tmp_path / "provider", tmp_path / "sdk", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.fixture
def small_process(tmp_path):
    process = tmp_path / "process"
    binary = process / "rootfs/usr/bin/python3.wasm"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixed interpreter")
    recipe = json.loads((assembly.PORT / "assembly_recipe.json").read_text())
    manifest = {
        "files": {"/usr/bin/python3.wasm": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "dynamic_abi": recipe["abi"],
        "recipe": {
            "version": "3.13.7",
            "target": recipe["target"],
            "prefix": "/usr",
            "target_profile": "wasi-cpython-v2",
        },
        "site_packages": "/usr/lib/python3.13/site-packages",
        "runtime_sources": {"dynamic.c": recipe["main_bridge_sha256"]},
        "process_port": {"recipe": {"patch_sha256": recipe["process_patch_sha256"]}},
    }
    (process / "manifest.json").write_text(json.dumps(manifest))
    return process, manifest


@pytest.mark.parametrize("field,value", [("target", "host-linux"), ("prefix", "/elsewhere")])
def test_assembly_rejects_wrong_target_or_prefix_without_changing_output(small_process, tmp_path, field, value):
    process, manifest = small_process
    manifest["recipe"][field] = value
    (process / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    with pytest.raises(ValueError, match="outside the pinned ctypes cohort"):
        assembly.build_runtime(process, tmp_path / "zlib", tmp_path / "libffi", tmp_path / "ctypes", output)
    assert (output / "sentinel").read_text() == "preserve"


def test_assembly_rejects_modified_input_and_copied_rootfs(small_process, tmp_path):
    process, manifest = small_process
    copied = tmp_path / "copied"
    shutil.copytree(process / "rootfs", copied / "rootfs")
    (copied / "rootfs/usr/bin/python3.wasm").write_bytes(b"altered during copy")
    with pytest.raises(ValueError, match="differs from its manifest"):
        assembly._verified_rootfs(copied, manifest)

    (process / "rootfs/usr/bin/python3.wasm").write_bytes(b"corrupt source")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    with pytest.raises(ValueError, match="differs from its manifest"):
        assembly.build_runtime(process, tmp_path / "zlib", tmp_path / "libffi", tmp_path / "ctypes", output)
    assert (output / "sentinel").read_text() == "preserve"
