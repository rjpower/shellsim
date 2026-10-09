"""Admit real sealed shared DAG metadata before staging dependency headers."""

import hashlib

import pytest

from ports._support.wasm import leb
from ports.native import imaging_shared
from ports.native.dependencies import seal_artifact, verify_artifact

TOOLCHAIN = {"sdk": "verified-fixture"}


@pytest.fixture
def graph(tmp_path):
    providers = {}
    manifests = {}

    def add(name, children=(), *, soname=None, header=None, toolchain=TOOLCHAIN):
        port = "native/" + name
        prefix = tmp_path / name
        (prefix / "lib").mkdir(parents=True)
        (prefix / "include").mkdir()
        soname = soname or ("lib" + name + ".so")
        needed = [manifests["native/" + child]["inputs"]["recipe"]["soname"] for child in children]
        payload = leb(len(needed)) + b"".join(leb(len(item.encode())) + item.encode() for item in needed)
        custom = leb(8) + b"dylink.0" + b"\x02" + leb(len(payload)) + payload
        (prefix / "lib" / soname).write_bytes(b"\0asm\x01\0\0\0\0" + leb(len(custom)) + custom)
        relative = "include/" + (header or (name + ".h"))
        (prefix / relative).write_bytes(name.encode())
        recipe = {
            "name": name,
            "version": "1.0",
            "target": "wasm32-wasip1",
            "target_profile": "wasi-cpython-v2",
            "abi": imaging_shared.ABI,
            "linkage": "shared",
            "soname": soname,
            "target_dependencies": [{"port": "native/" + child, "version": "1.0"} for child in children],
            "exports": {"headers": [relative], "libraries": ["lib/" + soname]},
            "source": {"sha256": hashlib.sha256(name.encode()).hexdigest()},
            "build_scripts": [],
        }
        inputs = {
            "recipe": recipe,
            "recipe_sha256": imaging_shared.digest(recipe),
            "source_sha256": recipe["source"]["sha256"],
            "toolchain": toolchain,
            "dependency_artifacts": {
                "native/" + child: manifests["native/" + child]["artifact_sha256"] for child in children
            },
        }
        manifests[port] = seal_artifact(prefix, inputs)
        providers[port] = prefix
        return prefix

    def consumer(*names):
        return {
            "target": "wasm32-wasip1",
            "target_profile": "wasi-cpython-v2",
            "abi": imaging_shared.ABI,
            "target_dependencies": [{"port": "native/" + name, "version": "1.0"} for name in names],
        }

    return add, consumer, providers, manifests


def test_deep_chain_stages_headers_but_preserves_direct_link_edges(tmp_path, graph):
    add, consumer, providers, _ = graph
    add("leaf")
    add("middle", ["leaf"])
    add("parent", ["middle"])
    selected, links = imaging_shared.shared_dependencies(consumer("parent"), providers, tmp_path / "stage", TOOLCHAIN)
    assert list(selected) == ["native/parent"]
    assert links == [providers["native/parent"] / "lib/libparent.so"]
    assert (tmp_path / "stage/include/leaf.h").read_bytes() == b"leaf"
    assert (tmp_path / "stage/include/middle.h").read_bytes() == b"middle"


def test_diamond_reuses_the_verified_provider(tmp_path, graph, monkeypatch):
    add, consumer, providers, _ = graph
    add("leaf")
    add("left", ["leaf"])
    add("right", ["leaf"])
    add("top", ["left", "right"])
    visited = []

    def verify(path):
        visited.append(path)
        return verify_artifact(path)

    monkeypatch.setattr(imaging_shared, "verify_artifact", verify)
    imaging_shared.shared_dependencies(consumer("top"), providers, tmp_path / "stage", TOOLCHAIN)
    assert visited.count(providers["native/leaf"]) == 1


@pytest.mark.parametrize("problem", ["cohort", "missing", "changed-edge", "unexpected-needed"])
def test_deep_failure_leaves_destination_uncreated(tmp_path, graph, problem):
    add, consumer, providers, manifests = graph
    add("leaf")
    add("middle", ["leaf"])
    add("top", ["middle"])
    add("good")
    if problem == "missing":
        del providers["native/leaf"]
    else:
        prefix = providers["native/leaf"]
        inputs = manifests["native/leaf"]["inputs"]
        if problem == "cohort":
            inputs["toolchain"] = {"sdk": "wrong"}
        elif problem == "changed-edge":
            (prefix / "include/leaf.h").write_bytes(b"updated provider")
        else:
            payload = b"\x01\x0bundeclared!"
            custom = leb(8) + b"dylink.0\x02" + leb(len(payload)) + payload
            (prefix / "lib/libleaf.so").write_bytes(b"\0asm\x01\0\0\0\0" + leb(len(custom)) + custom)
        seal_artifact(prefix, inputs)
    with pytest.raises(ValueError):
        imaging_shared.shared_dependencies(consumer("good", "top"), providers, tmp_path / "stage", TOOLCHAIN)
    assert not (tmp_path / "stage").exists()


def test_cycle_fails_before_staging(tmp_path, graph):
    add, consumer, providers, manifests = graph
    add("leaf")
    add("parent", ["leaf"])
    inputs = manifests["native/leaf"]["inputs"]
    inputs["recipe"]["target_dependencies"] = [{"port": "native/parent", "version": "1.0"}]
    inputs["recipe_sha256"] = imaging_shared.digest(inputs["recipe"])
    inputs["dependency_artifacts"] = {"native/parent": manifests["native/parent"]["artifact_sha256"]}
    seal_artifact(providers["native/leaf"], inputs)
    with pytest.raises(ValueError, match="Cyclic"):
        imaging_shared.shared_dependencies(consumer("parent"), providers, tmp_path / "stage", TOOLCHAIN)
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize("collision", ["header", "soname"])
def test_conflicting_provider_exports_fail_before_staging(tmp_path, graph, collision):
    add, consumer, providers, _ = graph
    add("left", header="common.h" if collision == "header" else None)
    add(
        "right",
        header="common.h" if collision == "header" else None,
        soname="libleft.so" if collision == "soname" else None,
    )
    with pytest.raises(ValueError, match="Conflicting"):
        imaging_shared.shared_dependencies(consumer("left", "right"), providers, tmp_path / "stage", TOOLCHAIN)
    assert not (tmp_path / "stage").exists()


def test_header_staging_rejects_redirected_directory(tmp_path, graph):
    add, consumer, providers, _ = graph
    add("leaf")
    destination = tmp_path / "stage"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "include").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="destination"):
        imaging_shared.shared_dependencies(consumer("leaf"), providers, destination, TOOLCHAIN)
    assert not (outside / "leaf.h").exists()
