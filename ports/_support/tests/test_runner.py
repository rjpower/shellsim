"""Exercise dependency rebuild propagation and cache rejection through real adapters."""

import hashlib
import json
import zipfile
from dataclasses import dataclass

import pytest

from ports._support import runner
from ports._support.cohort import Receipt
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

    recipe = {"name": "admitted-producer", "sdk": {"version": "34.0", "sha256": "7" * 64}}

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
    # This receipt-resolution fixture uses synthetic compiler/platform bytes;
    # real archive selection is covered by the runtime-profile link tests.
    monkeypatch.setattr(
        cohort_module,
        "host_executable_flags",
        lambda sysroot, target: (str(sysroot / "lib" / target / "libc.a"),),
    )
    resolved = resolved_toolchain(original, bootstrap_compiler, bootstrap_platform)
    target = runner._native_target(resolved)
    assert target.toolchain["compiler"] == bootstrap_compiler.sha256
    assert target.toolchain["platform"] == bootstrap_platform.sha256
    assert target.toolchain["sdk"] == bootstrap_platform.contents["identity"]["recipe"]["sdk"]
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


@dataclass(frozen=True)
class PlatformCohort:
    sysroot: Receipt


def test_executable_archive_admission_rejects_modified_and_unrecorded_files(tmp_path):
    path = tmp_path / "sysroot/lib/setjmp.a"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"!<arch>\nverified SDK archive")
    receipt = Receipt(
        tmp_path,
        tmp_path / "manifest.json",
        "0" * 64,
        {"artifacts": {"sysroot/lib/setjmp.a": hashlib.sha256(path.read_bytes()).hexdigest()}},
    )
    cohort = PlatformCohort(receipt)
    declared = {"executable_cohort_link_inputs": ["lib/setjmp.a"]}
    assert runner._executable_cohort_link_inputs(cohort, declared) == (path,)
    path.write_bytes(b"!<arch>\nmodified archive")
    with pytest.raises(ValueError, match="differs"):
        runner._executable_cohort_link_inputs(cohort, declared)
    unrecorded = path.with_name("ambient.a")
    unrecorded.write_bytes(b"!<arch>\nambient archive")
    with pytest.raises(ValueError, match="differs"):
        runner._executable_cohort_link_inputs(cohort, {"executable_cohort_link_inputs": ["lib/ambient.a"]})


@pytest.mark.parametrize("declared", [["../host.a"], ["/host.a"], ["lib/setjmp.a", "lib/setjmp.a"]])
def test_executable_archive_admission_rejects_escaping_or_duplicate_inputs(tmp_path, declared):
    path = tmp_path / "sysroot/lib/setjmp.a"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"!<arch>\nverified SDK archive")
    receipt = Receipt(
        tmp_path,
        tmp_path / "manifest.json",
        "0" * 64,
        {"artifacts": {"sysroot/lib/setjmp.a": hashlib.sha256(path.read_bytes()).hexdigest()}},
    )
    with pytest.raises(ValueError):
        runner._executable_cohort_link_inputs(PlatformCohort(receipt), {"executable_cohort_link_inputs": declared})


def test_publish_stdlib_runtime_provider_closes_pillow_dependencies(tmp_path):
    """Exercise real graph publication and loading when stdlib also owns zlib."""
    import os
    from pathlib import Path

    import shellsim
    from shellsim._cpython_universe import Universe

    from ports._support.cohort import load_cohort

    cohort_path = os.environ.get("SHELLSIM_PILLOW_GRAPH_COHORT")
    store_path = os.environ.get("SHELLSIM_PILLOW_GRAPH_STORE")
    if cohort_path is None or store_path is None:
        pytest.skip("real Pillow graph inputs were not supplied")
    cohort = load_cohort(Path(cohort_path))
    build = runner.build_graph(
        Path(__file__).parents[2], ["python/pillow/graph-recipe.json"], cohort, Path(store_path), offline=True
    )
    release = runner.publish_graph(build, cohort, tmp_path / "release")
    runtime = shellsim.CPythonRuntime.from_release(release, cache_dir=tmp_path / "cache")
    universe = Universe(runtime.universe, abi=cohort.dynamic_abi, python_version=cohort.python.version)
    providers = universe.provider_closure({"libfreetype.so"})
    assert "libz.so" in providers
    assert providers["libz.so"].read_bytes() == (runtime.bundle / "rootfs/lib/libz.so").read_bytes()
    environment = shellsim.Environment.from_release(release, pypi=["pillow==12.3.0"], cache_dir=tmp_path / "cache")
    environment.write_file(
        "/work/raster.py", b"from PIL import ImageFont\nassert sum(ImageFont.load_default(size=24).getmask('A')) > 0\n"
    )
    result = environment.run("python /work/raster.py")
    assert result.returncode == 0 and result.stop_reason is None, result.stderr


def test_profile_provider_selection_rebuilds_verified_wheel_consumers(tmp_path, monkeypatch):
    ports, store = tmp_path / "ports", tmp_path / "store"
    provider, provider_recipe = _wheel_port(ports, store, "provider", [])
    alternate = provider.with_name("alternate.json")
    alternate.write_text(json.dumps({**provider_recipe, "features": {"alternate-build": True}}))
    consumer, recipe = _wheel_port(ports, store, "consumer", ["provider"])
    consumer.write_text(json.dumps({**recipe, "build_profile": "pure-test"}))
    profile_path = ports / "profiles/pure-test.json"
    profile_path.parent.mkdir()
    profile = {
        "schema_version": 1,
        "target": PureCohort.target,
        "target_profile": PureCohort.dynamic_abi,
        "abi": PureCohort.dynamic_abi,
        "build_dependencies": [],
        "platform_dependencies": [],
        "dependency_recipes": {"python/provider": "python/provider/recipe.json"},
    }
    profile_path.write_text(json.dumps(profile))
    calls = []
    real_build = runner.build_pure_wheel

    def observed_build(request):
        calls.append(request.recipe["name"])
        return real_build(request)

    monkeypatch.setattr(runner, "build_pure_wheel", observed_build)
    first = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    receipt = json.loads((first.results["python/consumer/recipe.json"] / "build-receipt.json").read_text())
    assert receipt["inputs"]["build_profile"] == {
        "reference": "profiles/pure-test.json",
        "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    }
    calls.clear()
    cached = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert not calls
    assert cached.results == first.results
    profile["dependency_recipes"]["python/provider"] = "python/provider/alternate.json"
    profile_path.write_text(json.dumps(profile))
    changed = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert calls == ["provider", "consumer"]
    assert "python/provider/recipe.json" not in changed.results
    assert changed.results["python/consumer/recipe.json"] != first.results["python/consumer/recipe.json"]
    calls.clear()
    # Equal resolved values still retain the exact authored profile bytes.
    profile_path.write_text(json.dumps(profile, indent=2))
    reauthored = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert calls == ["consumer"]
    assert reauthored.results["python/provider/alternate.json"] == changed.results["python/provider/alternate.json"]
    assert reauthored.results["python/consumer/recipe.json"] != changed.results["python/consumer/recipe.json"]
    calls.clear()
    artifact = reauthored.results["python/consumer/recipe.json"] / "wheels/consumer-1-py3-none-any.whl"
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert not calls
