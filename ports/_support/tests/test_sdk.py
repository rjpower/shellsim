"""Bounded producer interfaces exercise missing SDK nodes and immutable reuse.

The compiler, libc, Python and resolver interfaces emit tiny real inventories;
the orchestration still fetches/extracts pinned sources and verifies product bytes.
This does not execute a cold LLVM bootstrap.
"""

import json
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports._support import sdk
from ports._support.graph import Graph, plan
from ports._support.sdk_products import Tool, file_hash, json_hash


@pytest.fixture
def product_graph(tmp_path, monkeypatch):
    store = tmp_path / "store"
    calls = []
    root = tmp_path / "archive-source" / "sdk"
    files = {
        "bin/" + name: name.encode()
        for name in ("clang", "clang++", "clang.cfg", "clang++.cfg", "llvm-ar", "llvm-ranlib", "llvm-nm", "llvm-strip")
    }
    files["lib/clang/23/include/stddef.h"] = b"typedef unsigned long size_t;\n"
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    archive = tmp_path / "sdk.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(root, arcname="sdk")
    source = {"url": "https://example.invalid/sdk.tar.gz", "filename": "sdk.tar.gz", "sha256": file_hash(archive)}
    cached = store / "sources" / source["sha256"] / source["filename"]
    cached.parent.mkdir(parents=True)
    cached.write_bytes(archive.read_bytes())
    tooling = {name: file_hash(root / name) for name in files}
    recipes = {
        "tooling": {"build_scripts": [], "sdk": source, "sdk_tooling_digest": json_hash(tooling)},
        "compiler": {"name": "fixture-compiler", "build_scripts": [], "source": source},
        "platform": {"name": "fixture-platform", "build_scripts": [], "sdk": source, "wasi_libc": source},
        "cpython": {"name": "fixture-python", "build_scripts": [], "source": source},
        "resolver": {"name": "fixture-resolver", "build_scripts": []},
    }
    adapters = {
        "tooling": "sdk-tooling",
        "compiler": "llvm-host",
        "platform": "wasi-sysroot",
        "cpython": "cpython-threaded",
        "resolver": "uv-host",
    }
    edges = {
        "tooling": (),
        "compiler": (),
        "platform": ("compiler", "tooling"),
        "cpython": ("compiler", "platform", "tooling"),
        "resolver": (),
    }
    nodes = []
    for name, recipe in recipes.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(recipe))
        nodes.append(sdk.SDKProduct(name, adapters[name], path, recipe, edges[name]))
    definition = sdk.SDKDefinition(
        "fixture", "sdks/fixture.json", {"target": "fixture-target", "abi": "fixture-abi"}, tuple(nodes)
    )
    monkeypatch.setattr(sdk, "definition", lambda *_args: definition)
    executable = tmp_path / "native-tool"
    executable.write_bytes(b"native-seed")
    tool = Tool(executable, file_hash(executable))
    seed = sdk.HostSeed(
        dict.fromkeys(("cmake", "ninja", "make"), tool), dict.fromkeys(("cc", "cxx", "cmake", "ninja"), tool), tool
    )
    monkeypatch.setattr(sdk, "load_seed", lambda _path: seed)

    def product(name, work):
        calls.append(name)
        work.mkdir(parents=True)
        if name == "compiler":
            for executable in ("clang", "clang++", "llvm-ar", "llvm-ranlib", "llvm-strip"):
                path = work / "bin" / executable
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(executable.encode())
        else:
            (work / "inventory").write_bytes(name.encode())
        inventory = {str(path.relative_to(work)): file_hash(path) for path in work.rglob("*") if path.is_file()}
        manifest = {"identity": {"recipe": recipes[name]}, "artifacts": inventory}
        (work / "manifest.json").write_text(json.dumps(manifest))
        return work

    def compiler(archive, cc, cxx, cmake, ninja, work):
        assert archive.read_bytes() == cached.read_bytes()
        assert (cc, cxx, cmake, ninja) == (tool.path,) * 4
        return product("compiler", work / "products" / str(len(list((work / "products").glob("*")))))

    def platform(archive, libc, compiler, cmake, ninja, work):
        assert (compiler / "manifest.json").exists()
        assert archive == libc == cached
        return product("platform", work)

    def cpython(archive, helper, tooling, platform, compiler, make, work):
        assert helper == make == tool.path
        assert (tooling / "lib/clang/23/include/stddef.h").is_file()
        assert (platform / "manifest.json").exists() and (compiler / "manifest.json").exists()
        work.mkdir(parents=True)
        calls.append("cpython")
        (work / "rootfs").mkdir()
        (work / "rootfs/python.wasm").write_bytes(b"python")
        manifest = {"recipe": recipes["cpython"], "files": {"/python.wasm": file_hash(work / "rootfs/python.wasm")}}
        (work / "manifest.json").write_text(json.dumps(manifest))
        return work

    def resolver(work, *, offline):
        assert not offline
        calls.append("resolver")
        work.mkdir(parents=True)
        (work / "uv").write_bytes(b"resolver")
        (work / "artifact.json").write_text(
            json.dumps(
                {
                    "recipe": recipes["resolver"],
                    "build": {"locked": True},
                    "executable": {"file": "uv", "sha256": file_hash(work / "uv")},
                }
            )
        )
        return work / "uv"

    monkeypatch.setattr("ports.toolchain.llvm.compiler.build", compiler)
    monkeypatch.setattr("ports.toolchain.wasi_threads.dynamic.build", platform)
    monkeypatch.setattr("ports.python.cpython.threaded.build", cpython)
    monkeypatch.setattr("ports.toolchain.uv.build.build", resolver)
    monkeypatch.setattr(
        sdk,
        "verify_cpython_recipe",
        lambda recipe: recipe == recipes["cpython"] or pytest.fail("wrong Python producer"),
    )
    monkeypatch.setattr(
        sdk,
        "admit_sdk",
        lambda tooling, compiler, platform, cpython, runtime, tools, entries, **metadata: SimpleNamespace(
            identity=json_hash([tooling.sha256, compiler.sha256, platform.sha256]),
            tools=tools,
            entries=entries,
            metadata=metadata,
        ),
    )
    return store, calls, definition, seed


def test_missing_products_use_existing_producers_and_repeat_without_build(product_graph):
    store, calls, _, _ = product_graph
    first = sdk.materialize(store, Graph((), ()), store, python=True, offline=False)
    assert calls == ["compiler", "platform", "cpython", "resolver"]
    assert first.metadata == {"target": "fixture-target", "abi": "fixture-abi"}
    assert first.entries["cc"]["path"] == "bin/clang"
    calls.clear()
    second = sdk.materialize(store, Graph((), ()), store, python=True, offline=False)
    assert not calls and second.identity == first.identity
    assert list((store / "materialized-sdks").glob("*.json"))


def test_native_materialization_omits_python_and_resolver_and_rejects_corruption(product_graph):
    store, calls, _, _ = product_graph
    sdk.materialize(store, Graph((), ()), store, offline=True)
    assert calls == ["compiler", "platform"]
    record = json.loads((store / "sdk-products/fixture/compiler.json").read_text())
    (Path(record["product"]["root"]) / "bin/clang").write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError):
        sdk.materialize(store, Graph((), ()), store, offline=True)
    assert not calls


def test_imported_compiler_remains_usable_when_seeding_missing_python(product_graph, monkeypatch):
    store, calls, definition, seed = product_graph
    sdk.materialize(store, Graph((), ()), store, offline=True)
    path = store / "sdk-products/fixture/compiler.json"
    record = json.loads(path.read_text())
    record["origin"] = {"kind": "verified-legacy-import", "descriptor_sha256": "a" * 64}
    record["inputs"] = {name: record["inputs"][name] for name in ("producer_policy", "dependencies")}
    path.write_text(json.dumps(record))
    calls.clear()
    sdk.materialize(store, Graph((), ()), store, python=True, offline=False)
    assert calls == ["cpython", "resolver"]
    compiler = next(node for node in definition.products if node.name == "compiler")
    inputs = sdk.node_inputs(compiler, {}, seed)
    changed = {**inputs, "producer_policy": "b" * 64}
    assert not sdk.compatible_inputs(record, changed)


def test_producer_policy_change_rebuilds_its_product_and_dependents(product_graph, monkeypatch):
    store, calls, definition, _ = product_graph
    sdk.materialize(store, Graph((), ()), store, offline=True)
    changed = tuple(
        replace(node, recipe={**node.recipe, "configuration": "changed"}) if node.name == "compiler" else node
        for node in definition.products
    )
    monkeypatch.setattr(sdk, "definition", lambda *_args: replace(definition, products=changed))
    calls.clear()
    # The fixture producer emits its old recipe, which must fail admission; the
    # changed input must reach that producer instead of failing on a stale hit.
    with pytest.raises(ValueError):
        sdk.materialize(store, Graph((), ()), store, offline=True)
    assert calls == ["compiler"]


def test_missing_host_compiler_fails_with_a_specific_prerequisite(product_graph, monkeypatch):
    store, calls, _, seed = product_graph
    monkeypatch.setattr(sdk, "load_seed", lambda _path: replace(seed, compiler_tools={}))
    with pytest.raises(sdk.PrerequisiteError, match="host compiler seed"):
        sdk.materialize(store, Graph((), ()), store, offline=True)
    assert not calls


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [("sha256", "0" * 64, "producer recipe pin"), ("dependencies", [], "producer dependencies")],
)
def test_changed_sdk_producer_pin_or_edge_fails_before_store_creation(tmp_path, monkeypatch, field, value, error):
    ports = Path(__file__).resolve().parents[2]
    graph = plan(ports, ["native/freetype/graph-recipe.json"])
    read_document = sdk._document

    def changed_document(root, reference):
        path, data, document = read_document(root, reference)
        if "products" in document:
            document["products"]["platform"][field] = value
        return path, data, document

    monkeypatch.setattr(sdk, "_document", changed_document)
    store = tmp_path / "store"
    with pytest.raises(ValueError, match=error):
        sdk.materialize(ports, graph, store, offline=True)
    assert not store.exists()


def test_changed_pinned_driver_fails_before_cached_product_use(product_graph, monkeypatch):
    store, calls, definition, _ = product_graph
    compiler = next(node for node in definition.products if node.name == "compiler")
    driver = compiler.recipe_path.parent / "fixture-driver.py"
    driver.write_text("PINNED = True\n")
    compiler.recipe["build_scripts"] = [{"file": driver.name, "sha256": file_hash(driver)}]
    sdk.materialize(store, Graph((), ()), store, offline=True)
    calls.clear()
    driver.write_text("PINNED = False\n")
    with pytest.raises(ValueError, match="Build script hash mismatch"):
        sdk.materialize(store, Graph((), ()), store, offline=True)
    assert not calls


def test_native_recipe_omission_selects_default_and_preserves_explicit_sdk(tmp_path):
    import shutil

    ports = Path(__file__).resolve().parents[2]
    shutil.copytree(ports / "sdks", tmp_path / "sdks")
    shutil.copytree(ports / "toolchain/llvm", tmp_path / "toolchain/llvm")
    shutil.copytree(ports / "toolchain/wasi_threads", tmp_path / "toolchain/wasi_threads")
    recipe = tmp_path / "native/example/recipe.json"
    recipe.parent.mkdir(parents=True)
    authored = {"name": "example", "version": "1", "build": {"adapter": "cmake"}}
    recipe.write_text(json.dumps(authored))
    implicit = plan(tmp_path, ["native/example"]).ports[-1]
    assert implicit.sdk_selection.reference == "sdks/wasi-threads-v3.json"
    assert {edge.kind for edge in implicit.dependencies} == {"build", "platform"}
    recipe.write_text(json.dumps({**authored, "sdk": "wasi-threads-v3"}))
    explicit = plan(tmp_path, ["native/example"], default_sdk="unavailable").ports[-1]
    assert explicit.sdk_selection == implicit.sdk_selection
    recipe.write_text(json.dumps({**authored, "sdk": "default"}))
    with pytest.raises(ValueError):
        plan(tmp_path, ["native/example"], default_sdk="unavailable")


def test_failed_nonresumable_producer_retries_in_a_fresh_attempt(product_graph, monkeypatch):
    store, calls, _, _ = product_graph
    from ports.toolchain.wasi_threads import dynamic

    actual = dynamic.build
    attempts = []

    def failing_once(archive, libc, compiler, cmake, ninja, work):
        attempts.append(work)
        if len(attempts) == 1:
            work.mkdir(parents=True)
            (work / "failure.log").write_text("interrupted fixture build")
            raise RuntimeError("fixture interruption")
        return actual(archive, libc, compiler, cmake, ninja, work)

    monkeypatch.setattr(dynamic, "build", failing_once)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        sdk.materialize(store, Graph((), ()), store, offline=True)
    sdk.materialize(store, Graph((), ()), store, offline=True)
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    assert (attempts[0] / "failure.log").read_text() == "interrupted fixture build"
    assert calls == ["compiler", "platform"]


def test_llvm_patch_update_reuses_the_producers_retained_workspace(product_graph, monkeypatch):
    store, calls, definition, _ = product_graph
    from ports.toolchain.llvm import compiler

    actual = compiler.build
    workspaces = []

    def observed(archive, cc, cxx, cmake, ninja, work):
        workspaces.append(work)
        return actual(archive, cc, cxx, cmake, ninja, work)

    monkeypatch.setattr(compiler, "build", observed)
    sdk.materialize(store, Graph((), ()), store, offline=True)
    original = json.loads((store / "sdk-products/fixture/compiler.json").read_text())
    node = next(node for node in definition.products if node.name == "compiler")
    node.recipe["patches"] = [{"file": "appended.patch", "sha256": "a" * 64}]
    calls.clear()
    sdk.materialize(store, Graph((), ()), store, offline=True)
    assert calls == ["compiler", "platform"]
    assert workspaces[0] == workspaces[1]
    # The real compiler producer owns compatibility admission; these bounded
    # interface products establish orchestration reuse and immutable old output.
    assert sdk.receipt(store, original["product"]).contents["identity"]["recipe"].get("patches") is None


def test_missing_offline_resolver_fails_before_its_network_producer(product_graph):
    store, calls, definition, seed = product_graph
    resolver = next(node for node in definition.products if node.name == "resolver")
    with pytest.raises(sdk.PrerequisiteError, match="offline resolver product"):
        sdk._produce(resolver, {}, seed, store, store / "unused", True)
    assert not calls and not (store / "unused").exists()
