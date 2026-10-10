"""Exercise dependency rebuild propagation and cache rejection through real adapters."""

import hashlib
import json
import zipfile
from dataclasses import dataclass

import pytest

from ports._support import runner
from ports._support.graph import Graph, Port


@dataclass(frozen=True)
class PureCohort:
    target: str = "wasm32-wasip1-threads"
    dynamic_abi: str = "test-pure-cohort"
    identity: str = "pinned-cohort"

    @property
    def toolchain_receipt(self):
        return {"cohort": self.identity, "target": self.target, "abi": self.dynamic_abi}


def _wheel_port(ports, store, name, dependencies):
    directory = ports / "python" / name
    directory.mkdir(parents=True)
    wheel = directory / f"{name}-1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{name}.py", "VALUE = 42\n")
        archive.writestr(f"{name}-1.dist-info/METADATA", f"Name: {name}\nVersion: 1\n")
        archive.writestr(f"{name}-1.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    source = store / "sources" / digest / wheel.name
    source.parent.mkdir(parents=True)
    source.write_bytes(wheel.read_bytes())
    recipe = {
        "name": name,
        "version": "1",
        "build": {"adapter": "pure-wheel"},
        "source": {"filename": wheel.name, "url": "https://example.invalid/" + wheel.name, "sha256": digest},
        "target_dependencies": [{"port": "python/" + dependency, "version": "1"} for dependency in dependencies],
    }
    path = directory / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path, recipe


def test_graph_reuses_verified_builds_and_rebuilds_dependents(tmp_path, monkeypatch):
    ports, store = tmp_path / "ports", tmp_path / "store"
    provider, recipe = _wheel_port(ports, store, "provider", [])
    _wheel_port(ports, store, "consumer", ["provider"])
    calls = []
    real_build = runner.build_pure_wheel

    def observed_build(request):
        calls.append(request.recipe["name"])
        return real_build(request)

    monkeypatch.setattr(runner, "build_pure_wheel", observed_build)
    first = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert calls == ["provider", "consumer"]
    calls.clear()
    second = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert not calls
    assert second.results == first.results

    provider.write_text(json.dumps({**recipe, "features": {"new-build-input": True}}))
    third = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert calls == ["provider", "consumer"]
    assert third.results["python/consumer/recipe.json"] != first.results["python/consumer/recipe.json"]

    artifact = third.results["python/provider/recipe.json"] / "wheels/provider-1-py3-none-any.whl"
    artifact.write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError):
        runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert not calls


def test_build_cache_ignores_acceptance_code_but_tracks_selected_adapter(tmp_path, monkeypatch):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    calls = []
    real_build = runner.build_pure_wheel

    def observed_build(request):
        calls.append(request.recipe["name"])
        return real_build(request)

    monkeypatch.setattr(runner, "build_pure_wheel", observed_build)
    runner.build_graph(ports, ["python/example"], PureCohort(), store, offline=True)
    assert calls == ["example"]

    real_hash = runner.file_hash

    def changed_acceptance(path, **kwargs):
        return "a" * 64 if path.name == "acceptance.py" else real_hash(path, **kwargs)

    monkeypatch.setattr(runner, "file_hash", changed_acceptance)
    calls.clear()
    runner.build_graph(ports, ["python/example"], PureCohort(), store, offline=True)
    assert not calls

    def changed_adapter(path, **kwargs):
        return "b" * 64 if path.name == "pure_wheel.py" else real_hash(path, **kwargs)

    monkeypatch.setattr(runner, "file_hash", changed_adapter)
    runner.build_graph(ports, ["python/example"], PureCohort(), store, offline=True)
    assert calls == ["example"]


def test_acceptance_uses_port_namespace_when_names_collide(tmp_path, monkeypatch):
    from ports._support import acceptance, native_artifacts
    from ports.native import dependencies

    native = Port("native/example/recipe.json", tmp_path, "example", "1", "a" * 64, (), {})
    python = Port("python/example/recipe.json", tmp_path, "example", "1", "b" * 64, (), {})
    native_result = tmp_path / "native-result"
    (native_result / "native").mkdir(parents=True)
    build = runner.GraphBuild(
        Graph((native.reference, python.reference), (native, python)),
        {native.reference: native_result, python.reference: tmp_path / "python-result"},
        {native.reference: PureCohort(), python.reference: PureCohort()},
    )
    kinds = []

    def observed_accept(request):
        kinds.append((request.port.reference, request.install_kind))
        request.output.mkdir(parents=True)

    monkeypatch.setattr(acceptance, "accept_port", observed_accept)
    monkeypatch.setattr(native_artifacts, "merge_dependency_sysroot", lambda *_args: None)
    monkeypatch.setattr(dependencies, "verify_artifact", lambda *_args: {})
    descriptor = tmp_path / "release.json"
    descriptor.write_text("{}")
    runner.accept_graph(build, PureCohort(), descriptor, tmp_path / "proof")
    assert kinds == [(native.reference, "native"), (python.reference, "pypi")]


def test_bootstrap_products_bind_native_metadata_and_acceptance(tmp_path, monkeypatch):
    from ports._support import acceptance, native_artifacts
    from ports._support import cohort as cohort_module
    from ports._support.acceptance import _native_command
    from ports._support.cohort import BuildCohort, Receipt, resolved_toolchain
    from ports.native import dependencies

    recipe = {"name": "admitted-producer"}

    def product(name, digest):
        root = tmp_path / name
        (root / "bin").mkdir(parents=True)
        for tool in ("clang", "clang++", "llvm-ar", "llvm-ranlib", "llvm-strip"):
            (root / "bin" / tool).write_bytes(name.encode())
        return Receipt(root, root / "manifest.json", digest * 64, {"identity": {"recipe": recipe}})

    original_compiler = product("original-compiler", "a")
    original_platform = product("original-platform", "b")
    sdk = product("sdk", "c")
    original = BuildCohort(sdk, original_compiler, original_platform, None, None, None, {}, {}, "d" * 64, True)
    bootstrap_compiler = product("bootstrap-compiler", "e")
    bootstrap_platform = product("bootstrap-platform", "f")
    monkeypatch.setattr(cohort_module, "local_recipe", lambda _name: recipe)
    resolved = resolved_toolchain(original, bootstrap_compiler, bootstrap_platform)
    target = runner._native_target(resolved)
    assert target.toolchain["compiler"] == bootstrap_compiler.sha256
    assert target.toolchain["platform"] == bootstrap_platform.sha256
    assert target.toolchain["cohort"] == resolved.identity != original.identity

    port = Port("native/example/recipe.json", tmp_path, "example", "1", "1" * 64, (), {"tests": [{"kind": "native"}]})
    result = tmp_path / "result"
    (result / "native").mkdir(parents=True)
    graph = runner.GraphBuild(Graph((port.reference,), (port,)), {port.reference: result}, {port.reference: resolved})
    commands, targets = [], []

    def observed_merge(_direct, _closure, prefix, actual_target):
        targets.append(actual_target.toolchain)
        prefix.mkdir(parents=True)

    def observed_accept(request):
        commands.append(
            _native_command(
                request.cohort, request.dependency_sysroot, tmp_path / "probe.c", tmp_path / "probe.wasm", [], [], []
            )
        )
        request.output.mkdir(parents=True)

    monkeypatch.setattr(native_artifacts, "merge_dependency_sysroot", observed_merge)
    monkeypatch.setattr(dependencies, "verify_artifact", lambda _prefix: {})
    monkeypatch.setattr(acceptance, "accept_port", observed_accept)
    (tmp_path / "release.json").write_text("{}")
    runner.accept_graph(graph, original, tmp_path / "release.json", tmp_path / "proof")
    assert targets == [target.toolchain]
    command = commands[0]
    assert command[0] == str(bootstrap_compiler.root / "bin/clang")
    assert "--sysroot=" + str(bootstrap_platform.root / "sysroot") in command
    assert not any(
        str(original_compiler.root) in argument or str(original_platform.root) in argument for argument in command
    )
    proof = json.loads((tmp_path / "proof/graph.json").read_text())
    assert proof["resolved_toolchains"][port.reference] == target.toolchain
