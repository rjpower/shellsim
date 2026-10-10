"""Reject incompatible or corrupted assembly inputs before publication."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ports._support.graph import plan
from ports._support.native_artifacts import NativeTarget
from ports._support.runtime_files import _verified_rootfs
from ports._support.sdk_products import file_hash
from ports.python.cpython.graph_assembly import assemble_stdlib


@pytest.fixture
def small_process(tmp_path):
    process = tmp_path / "process"
    binary = process / "rootfs/usr/bin/python3.wasm"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixed interpreter")
    recipe = next(
        port.recipe
        for port in plan(Path(__file__).parents[4] / "ports", ["python/cpython:stdlib-ctypes"]).ports
        if port.variant == "stdlib-ctypes"
    )
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
    with pytest.raises(ValueError):
        assemble_stdlib(
            process,
            {},
            {},
            NativeTarget("wasm32-wasip1-threads", "runtime", manifest["dynamic_abi"], {}),
            output,
            runtime_manifest_sha256=file_hash(process / "manifest.json"),
        )
    assert (output / "sentinel").read_text() == "preserve"


def test_assembly_rejects_modified_input_and_copied_rootfs(small_process, tmp_path):
    process, manifest = small_process
    copied = tmp_path / "copied"
    shutil.copytree(process / "rootfs", copied / "rootfs")
    (copied / "rootfs/usr/bin/python3.wasm").write_bytes(b"altered during copy")
    with pytest.raises(ValueError, match="differs from its manifest"):
        _verified_rootfs(copied, manifest)

    (process / "rootfs/usr/bin/python3.wasm").write_bytes(b"corrupt source")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    with pytest.raises(ValueError, match="differs from its manifest"):
        _verified_rootfs(process, manifest)
    assert (output / "sentinel").read_text() == "preserve"
