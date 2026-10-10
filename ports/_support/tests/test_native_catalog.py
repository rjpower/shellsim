"""Exercise public native installation, aliasing and sealed dependency rejection."""

import dataclasses
import json

import pytest

from ports._support.graph import Graph, Port
from ports._support.native_catalog import publish_native_catalog
from ports.native.dependencies import digest, seal_artifact


def _artifact(
    tmp_path, name, dependencies=(), runtime=(), aliases=None, target="wasm32-wasip1-threads", directories=None
):
    result = tmp_path / name
    prefix = result / "native"
    prefix.mkdir(parents=True)
    path = prefix / "bin" / name
    path.parent.mkdir()
    path.write_bytes(b"#!/bin/sh\nprintf installed\n")
    recipe = {
        "name": name,
        "version": "1",
        "target": target,
        "target_profile": "fixture-v3",
        "abi": "fixture-v3",
        "exports": {"tools": ["bin/" + name]},
        "target_dependencies": [{"port": "native/" + item.name, "version": item.version} for item in dependencies],
        "runtime_dependencies": [{"port": "native/" + item.name, "version": item.version} for item in runtime],
    }
    recipe["role"] = "guest-tool"
    if directories:
        recipe["empty_directories"] = directories
        for directory, mode in directories.items():
            (prefix / directory).mkdir(parents=True)
            (prefix / directory).chmod(mode)
    if aliases:
        recipe["install"] = {"name": aliases}
    inputs = {
        "recipe": recipe,
        "recipe_sha256": digest(recipe),
        "toolchain": {"cohort": "a" * 64, "target": recipe["target"], "abi": "fixture-v3"},
        "dependency_artifacts": {
            "native/" + item.name: json.loads((tmp_path / item.name / "native/artifact.json").read_text())[
                "artifact_sha256"
            ]
            for item in dependencies
        },
        "runtime_artifacts": {
            "native/" + item.name: json.loads((tmp_path / item.name / "native/artifact.json").read_text())[
                "artifact_sha256"
            ]
            for item in runtime
        },
    }
    seal_artifact(prefix, inputs)
    from ports._support.graph import Dependency

    edges = tuple(
        Dependency("native/" + item.name, "1", item.reference, kind)
        for kind, items in [("target", dependencies), ("runtime", runtime)]
        for item in items
    )
    declaration = recipe
    return Port("native/" + name + "/recipe.json", tmp_path, name, "1", digest(declaration), edges, declaration), result


def test_published_runtime_closure_installs_commands_and_rejects_edge_tamper(tmp_path):
    from shellsim import Environment
    from shellsim.native_packages import _NativePackageUniverse

    provider, provider_result = _artifact(tmp_path, "provider", target="wasm32-wasip1")
    consumer, consumer_result = _artifact(tmp_path, "consumer", runtime=(provider,), aliases="consumer-tool")
    host = dataclasses.replace(provider, reference="toolchain/host/recipe.json", recipe={"role": "host-tool"})
    graph = Graph((consumer.reference,), (host, provider, consumer))
    results = {
        host.reference: tmp_path / "does-not-exist",
        provider.reference: provider_result,
        consumer.reference: consumer_result,
    }
    catalog = publish_native_catalog(graph, results, tmp_path / "catalog")
    env = Environment()
    assert _NativePackageUniverse(catalog).install(env, "consumer-tool") == {"consumer-tool": "1", "provider": "1"}
    result = env.run("consumer; provider")
    assert result.returncode == 0
    assert result.stdout == b"installedinstalled"
    assert env.run("test -x /usr/local/bin/consumer").returncode == 0
    value = json.loads(catalog.read_text())
    record = next(item for item in value["packages"] if item["name"] == "consumer-tool")
    record["dependencies"][0]["linked"] = True
    catalog.write_text(json.dumps(value))
    fresh = Environment()
    fresh.write_file("/work/sentinel", b"preserved")
    with pytest.raises(ValueError):
        _NativePackageUniverse(catalog).install(fresh, "consumer-tool")
    assert fresh.read_file("/work/sentinel") == b"preserved"
    assert fresh.run("test ! -e /usr/local/bin/consumer").returncode == 0


def test_graph_publication_rejects_changed_install_recipe(tmp_path):
    port, result = _artifact(tmp_path, "example")
    changed = dataclasses.replace(
        port, recipe={**port.recipe, "install": {"destinations": {"bin/example": "/usr/bin/other"}}}
    )
    with pytest.raises(ValueError, match="different graph recipe"):
        publish_native_catalog(
            Graph((changed.reference,), (changed,)), {changed.reference: result}, tmp_path / "catalog"
        )


def test_linked_provider_cannot_cross_target_triples(tmp_path):
    provider, provider_result = _artifact(tmp_path, "serial-provider", target="wasm32-wasip1")
    consumer, consumer_result = _artifact(tmp_path, "threaded-consumer", dependencies=(provider,))
    graph = Graph((consumer.reference,), (provider, consumer))
    with pytest.raises(ValueError):
        publish_native_catalog(
            graph, {provider.reference: provider_result, consumer.reference: consumer_result}, tmp_path / "catalog"
        )


def test_public_install_preserves_and_restores_empty_sdk_directory(tmp_path):
    from shellsim import Environment
    from shellsim.native_packages import _NativePackageUniverse

    port, result = _artifact(tmp_path, "sdk", directories={"wasi-sysroot/include/c++/v1": 0o755})
    catalog = publish_native_catalog(Graph((port.reference,), (port,)), {port.reference: result}, tmp_path / "catalog")
    universe = _NativePackageUniverse(catalog)
    env = Environment()
    universe.install(env, "sdk")
    directory = "/usr/local/wasi-sysroot/include/c++/v1"
    assert env.run("test -d " + directory).returncode == 0
    assert env.run("rmdir " + directory).returncode == 0
    universe.install(env, "sdk")
    assert env.run("test -d " + directory).returncode == 0
    assert env.run("sdk").stdout == b"installed"


def test_native_transport_preserves_empty_directories_only_when_admitted(tmp_path):
    from shellsim._cpython_release import _extract
    from shellsim.cpython import PackageInstallError

    from ports.python.cpython.release import _archive

    source = tmp_path / "source"
    (source / "native/sdk/include/c++/v1").mkdir(parents=True)
    archive = tmp_path / "native.zip"
    _archive(source, archive, empty_directories=True)
    destination = tmp_path / "unpacked"
    _extract(archive, destination, allowed_roots=frozenset({"native"}), allow_empty_directories=True)
    directory = destination / "native/sdk/include/c++/v1"
    assert directory.is_dir()
    assert list(directory.iterdir()) == []
    with pytest.raises(PackageInstallError):
        _extract(archive, tmp_path / "unadmitted", allowed_roots=frozenset({"native"}))
