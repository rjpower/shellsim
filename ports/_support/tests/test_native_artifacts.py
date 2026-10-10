"""Exercise expanded exports, contained links and atomic dependency closure admission."""

from dataclasses import replace

import pytest

from ports._support.native_artifacts import NativeArtifact, NativeTarget, merge_dependency_sysroot, seal_native_install
from ports.native.dependencies import digest, seal_artifact, verify_artifact


@pytest.fixture
def target():
    return NativeTarget("wasm32-wasip1", "test-profile", None, {"compiler": "verified"})


def recipe(name, target, requirements=()):
    return {
        "name": name,
        "version": "1",
        "target": target.target,
        "target_profile": target.profile,
        "source": {"sha256": "0" * 64},
        "build_scripts": [],
        "target_dependencies": list(requirements),
        "exports": {"headers": ["include/" + name + ".h"]},
    }


def artifact(tmp_path, name, target, children=(), data=b"header"):
    prefix = tmp_path / name
    prefix.mkdir()
    r = recipe(name, target, [{"port": child, "version": "1"} for child, _ in children])
    path = prefix / r["exports"]["headers"][0]
    path.parent.mkdir()
    path.write_bytes(data)
    seal_artifact(
        prefix,
        {
            "recipe": r,
            "recipe_sha256": digest(r),
            "toolchain": dict(target.toolchain),
            "source_sha256": r["source"]["sha256"],
            "dependency_artifacts": {name: child.manifest["artifact_sha256"] for name, child in children},
        },
    )
    return NativeArtifact(prefix, verify_artifact(prefix))


def test_directory_expansion_and_contained_link_snapshot(tmp_path, target):
    staging = tmp_path / "stage"
    payload = staging / "usr/local"
    (payload / "include").mkdir(parents=True)
    (payload / "include/value.h").write_bytes(b"declaration")
    (payload / "lib").mkdir()
    (payload / "lib/value.a.1").write_bytes(b"archive")
    (payload / "lib/value.a").symlink_to("/usr/local/lib/value.a.1")
    original = recipe("value", target)
    original["exports"] = {"archives": ["lib/value.a"]}
    original["export_directories"] = {"headers": ["include"]}
    result = seal_native_install(original, tmp_path, staging, tmp_path / "output", target, {}, {})
    assert (result.prefix / "lib/value.a").read_bytes() == b"archive"
    assert not (result.prefix / "lib/value.a").is_symlink()
    assert result.manifest["inputs"]["graph_recipe"] == original
    assert result.manifest["inputs"]["recipe"]["exports"]["headers"] == ["include/value.h"]
    assert result.manifest["inputs"]["graph_recipe_sha256"] != result.manifest["inputs"]["recipe_sha256"]


@pytest.mark.parametrize("link", ["/etc/passwd", "../../outside", "missing"])
def test_invalid_install_link_does_not_publish(tmp_path, target, link):
    payload = tmp_path / "stage/usr/local/include"
    payload.mkdir(parents=True)
    (payload / "value.h").symlink_to(link)
    with pytest.raises(ValueError):
        seal_native_install(recipe("value", target), tmp_path, tmp_path / "stage", tmp_path / "output", target, {}, {})
    assert not (tmp_path / "output").exists()


def test_diamond_closure_merges_only_reachable_results(tmp_path, target):
    leaf = artifact(tmp_path, "leaf", target)
    left = artifact(tmp_path, "left", target, [("leaf", leaf)])
    right = artifact(tmp_path, "right", target, [("leaf", leaf)])
    unused = artifact(tmp_path, "unused", target)
    result = merge_dependency_sysroot(
        {"left": left, "right": right}, {"leaf": leaf, "unused": unused}, tmp_path / "sysroot", target
    )
    assert (result / "usr/local/include/leaf.h").read_bytes() == b"header"
    assert not (result / "usr/local/include/unused.h").exists()


def test_tampered_transitive_provider_does_not_stage(tmp_path, target):
    leaf = artifact(tmp_path, "leaf", target)
    parent = artifact(tmp_path, "parent", target, [("leaf", leaf)])
    (leaf.prefix / "include/leaf.h").write_bytes(b"changed")
    with pytest.raises(ValueError):
        merge_dependency_sysroot({"parent": parent}, {"leaf": leaf}, tmp_path / "sysroot", target)
    assert not (tmp_path / "sysroot").exists()


def test_cohort_mismatch_does_not_stage(tmp_path, target):
    item = artifact(tmp_path, "item", target)
    with pytest.raises(ValueError):
        merge_dependency_sysroot({"item": item}, {}, tmp_path / "sysroot", replace(target, profile="wrong"))
    assert not (tmp_path / "sysroot").exists()


def test_conflicting_shared_header_does_not_stage(tmp_path, target):
    first = artifact(tmp_path, "first", target)
    second = artifact(tmp_path, "second", target)
    for item, contents in ((first, b"one"), (second, b"two")):
        r = item.manifest["inputs"]["recipe"]
        old = item.prefix / r["exports"]["headers"][0]
        old.unlink()
        (item.prefix / "include/common.h").write_bytes(contents)
        r["exports"]["headers"] = ["include/common.h"]
        inputs = dict(item.manifest["inputs"])
        inputs["recipe_sha256"] = digest(r)
        seal_artifact(item.prefix, inputs)
    first = NativeArtifact(first.prefix, verify_artifact(first.prefix))
    second = NativeArtifact(second.prefix, verify_artifact(second.prefix))
    with pytest.raises(ValueError):
        merge_dependency_sysroot({"first": first, "second": second}, {}, tmp_path / "sysroot", target)
    assert not (tmp_path / "sysroot").exists()


@pytest.mark.parametrize("wrong_marker", [True, False])
def test_shared_library_rejection_leaves_output_absent(tmp_path, target, wrong_marker):
    from ports._support.wasm import leb, mark_abi

    target = replace(target, abi="declared-abi")
    payload = tmp_path / "stage/usr/local/lib"
    payload.mkdir(parents=True)
    library = payload / "libvalue.so"
    data = b"\0asm\x01\0\0\0"
    if not wrong_marker:
        name = b"dylink.0"
        needed = leb(1) + leb(11) + b"libother.so"
        section = leb(len(name)) + name + bytes([2]) + leb(len(needed)) + needed
        data += b"\0" + leb(len(section)) + section
    library.write_bytes(data)
    if wrong_marker:
        mark_abi(library, b"wrong-abi")
    original = recipe("value", target)
    original["abi"] = target.abi
    original["exports"] = {"shared_libraries": ["lib/libvalue.so"]}
    with pytest.raises(ValueError):
        seal_native_install(original, tmp_path, tmp_path / "stage", tmp_path / "output", target, {}, {})
    assert not (tmp_path / "output").exists()


def test_symlinked_install_parent_is_rejected_without_reading_target(tmp_path, target):
    outside = tmp_path / "outside/local/include"
    outside.mkdir(parents=True)
    (outside / "value.h").write_bytes(b"must not be admitted")
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "usr").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError):
        seal_native_install(recipe("value", target), tmp_path, staging, tmp_path / "output", target, {}, {})
    assert not (tmp_path / "output").exists()


def test_runtime_dependencies_are_verified_without_becoming_link_inputs(tmp_path, target):
    runtime = artifact(tmp_path, "runtime", target)
    consumer = recipe("consumer", target)
    consumer["runtime_dependencies"] = [{"port": "native/runtime", "version": "1"}]
    payload = tmp_path / "stage/usr/local/include"
    payload.mkdir(parents=True)
    (payload / "consumer.h").write_bytes(b"consumer")
    result = seal_native_install(
        consumer, tmp_path, tmp_path / "stage", tmp_path / "output", target, {}, {}, {"native/runtime": runtime}
    )
    assert result.manifest["inputs"]["dependency_artifacts"] == {}
    assert result.manifest["inputs"]["runtime_artifacts"] == {"native/runtime": runtime.manifest["artifact_sha256"]}
    consumer["runtime_dependencies"][0]["version"] = "2"
    with pytest.raises(ValueError, match="runtime dependency version"):
        seal_native_install(
            consumer, tmp_path, tmp_path / "stage", tmp_path / "invalid", target, {}, {}, {"native/runtime": runtime}
        )


def test_empty_directory_survives_sealing_and_dependency_merge(tmp_path, target):
    payload = tmp_path / "stage/usr/local"
    (payload / "include/c++/v1").mkdir(parents=True)
    (payload / "include/value.h").write_bytes(b"header")
    original = recipe("value", target)
    original["empty_directories"] = {"include/c++/v1": 0o755}
    result = seal_native_install(original, tmp_path, tmp_path / "stage", tmp_path / "output", target, {}, {})
    merged = merge_dependency_sysroot({"native/value": result}, {}, tmp_path / "sysroot", target)
    directory = merged / "usr/local/include/c++/v1"
    assert directory.is_dir()
    assert list(directory.iterdir()) == []
    (result.prefix / "include/c++/v1/injected").write_bytes(b"unexpected")
    with pytest.raises(ValueError):
        verify_artifact(result.prefix)


def test_wasm_tool_is_marked_and_wrong_existing_marker_rejected(tmp_path, target):
    from ports._support.native_artifacts import _abi
    from ports._support.wasm import mark_abi

    target = replace(target, abi="tool-abi")
    payload = tmp_path / "stage/usr/local/bin"
    payload.mkdir(parents=True)
    (payload / "tool").write_bytes(b"\0asm\x01\0\0\0")
    (payload / "script").write_bytes(b"#!/bin/sh\nprintf ok\n")
    original = recipe("value", target)
    original["abi"] = target.abi
    original["exports"] = {"tools": ["bin/tool", "bin/script"]}
    result = seal_native_install(original, tmp_path, tmp_path / "stage", tmp_path / "output", target, {}, {})
    assert _abi((result.prefix / "bin/tool").read_bytes()) == target.abi
    assert (result.prefix / "bin/script").read_bytes() == (payload / "script").read_bytes()
    mark_abi(payload / "tool", b"wrong")
    with pytest.raises(ValueError, match="ABI marker"):
        seal_native_install(original, tmp_path, tmp_path / "stage", tmp_path / "invalid", target, {}, {})
    assert not (tmp_path / "invalid").exists()
