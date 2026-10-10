"""Check real sealed stdlib overlay preservation and failed-publication atomicity."""

import json
import os
import shutil
from pathlib import Path

import pytest

from ports._support.native_artifacts import NativeArtifact, NativeTarget
from ports.native.dependencies import file_hash, verify_artifact
from ports.python.cpython.graph_assembly import assemble_stdlib


@pytest.fixture
def graph_inputs():
    names = ("SHELLSIM_GRAPH_STDLIB_RUNTIME", "SHELLSIM_GRAPH_STDLIB_MODULE", "SHELLSIM_GRAPH_STDLIB_PROVIDER")
    if any(name not in os.environ for name in names):
        pytest.skip("real stdlib graph artifacts were not supplied")
    runtime, module_path, provider_path = (Path(os.environ[name]) for name in names)
    module = NativeArtifact(module_path, verify_artifact(module_path))
    provider = NativeArtifact(provider_path, verify_artifact(provider_path))
    recipe = module.manifest["inputs"]["recipe"]
    target = NativeTarget(
        recipe["target"], recipe["target_profile"], recipe["abi"], module.manifest["inputs"]["toolchain"]
    )
    return runtime, module, provider, target


def test_real_stdlib_overlay_preserves_image_and_rejects_conflicting_destination(tmp_path, graph_inputs):
    runtime, module, provider, target = graph_inputs
    modules = {"python/cpython-stdlib-zlib": module}
    closure = {"native/zlib": provider}
    before = (runtime / "rootfs/usr/bin/python3.wasm").read_bytes()
    output, providers = assemble_stdlib(
        runtime,
        modules,
        closure,
        target,
        tmp_path / "assembled",
        runtime_manifest_sha256=file_hash(runtime / "manifest.json"),
    )
    assert (output / "rootfs/usr/bin/python3.wasm").read_bytes() == before
    assert (runtime / "rootfs/usr/bin/python3.wasm").read_bytes() == before
    assert file_hash(output / "rootfs/usr/lib/python3.13/lib-dynload/zlib.so") == file_hash(
        module.prefix / "lib-dynload/zlib.so"
    )
    assert "libz.so" in providers

    conflicting = tmp_path / "conflicting-base"
    shutil.copytree(runtime, conflicting)
    path = conflicting / "rootfs/usr/lib/python3.13/lib-dynload/zlib.so"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"existing different module")
    manifest = json.loads((conflicting / "manifest.json").read_bytes())
    manifest["files"]["/usr/lib/python3.13/lib-dynload/zlib.so"] = file_hash(path)
    (conflicting / "manifest.json").write_text(json.dumps(manifest))
    failed = tmp_path / "must-not-publish"
    with pytest.raises(ValueError, match="cannot replace"):
        assemble_stdlib(
            conflicting,
            modules,
            closure,
            target,
            failed,
            runtime_manifest_sha256=file_hash(conflicting / "manifest.json"),
        )
    assert not failed.exists()
    assert path.read_bytes() == b"existing different module"
