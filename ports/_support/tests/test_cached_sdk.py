"""Admit tiny retained inventories through real receipt and SDK verification.

Reuse the portable fixture's synthetic producer policies, while keeping input,
dependency, seed, inventory and typed SDK admission checks real. No producer runs.
"""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports._support import sdk
from ports._support.graph import Graph
from ports._support.sdk_products import MaterializedSDK, Tool, json_hash, tool_reference
from ports.buildomatic.tests.test_portable import proof, write
from ports.buildomatic.tests.test_portable import sdk_fixture as sdk_fixture


@pytest.fixture
def cached_sdk(tmp_path, monkeypatch, request):
    context = request.getfixturevalue("sdk_fixture")(python=True, host_python=True)
    ports = Path(__file__).resolve().parents[2]
    definition = sdk.definition(ports, Graph((), ()))
    nodes = tuple(
        replace(node, recipe={**node.recipe, "sdk_tooling_digest": json_hash(context.sdk.contents)})
        if node.name == "tooling"
        else node
        for node in definition.products
    )
    definition = replace(
        definition,
        name="cached-fixture",
        metadata={"target": context.target, "abi": context.dynamic_abi},
        products=nodes,
    )
    monkeypatch.setattr(sdk, "definition", lambda *_args: definition)
    monkeypatch.setattr(sdk, "verify_policy", lambda *_args: None)
    store = tmp_path / "store"
    index = store / "sdk-products" / definition.name
    index.mkdir(parents=True)
    tools = {**context.host_tools, **dict.fromkeys(("make", "cmake", "ninja"), context.host_tools["sh"])}
    native = {"path": str(tools["sh"].path), "sha256": tools["sh"].sha256}
    seed_path = index / "host-seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "host": "linux-x86_64",
                "tools": {name: tool_reference(tool) for name, tool in tools.items()},
                "compiler_tools": dict.fromkeys(("cc", "cxx", "cmake", "ninja"), native),
                "python_helper": native,
            }
        )
    )
    seed = sdk.load_seed(seed_path)
    resolver_root = tmp_path / "original-resolver"
    executable = write(resolver_root, "uv", b"fixture-uv", 0o755)
    resolver_node = next(node for node in nodes if node.name == "resolver")
    resolver = proof(
        resolver_root,
        {
            "recipe": resolver_node.recipe,
            "build": {"locked": True},
            "executable": {"file": "uv", "sha256": sdk.file_hash(executable)},
        },
        "artifact.json",
    )
    products = {
        "tooling": context.sdk,
        "compiler": context.llvm,
        "platform": context.sysroot,
        "cpython": context.cpython_manifest,
        "resolver": resolver,
    }
    admitted = {}
    for node in nodes:
        record = {
            "schema_version": 1,
            "origin": "producer",
            "inputs": sdk.node_inputs(node, admitted, seed),
            "product": sdk.product_reference(products[node.name]),
        }
        (index / (node.name + ".json")).write_text(json.dumps(record))
        admitted[node.name] = products[node.name]
    return SimpleNamespace(
        ports=ports, graph=Graph((), ()), store=store, index=index, definition=definition, seed=seed, products=products
    )


@pytest.fixture
def readonly_admission(monkeypatch):
    @contextmanager
    def guard():
        def reject(*args, **kwargs):
            pytest.fail("cached SDK admission attempted production or mutation")

        original_open = Path.open

        def open_readonly(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                reject()
            return original_open(path, mode, *args, **kwargs)

        with monkeypatch.context() as guarded:
            for name in (
                "materialize",
                "_materialize",
                "_produce",
                "producer_work",
                "_publish_json",
                "build_port",
                "fetch",
            ):
                guarded.setattr(sdk, name, reject)
            guarded.setattr(sdk.fcntl, "flock", reject)
            for name in ("mkdir", "write_text", "write_bytes", "touch", "rename", "replace", "unlink", "symlink_to"):
                guarded.setattr(Path, name, reject)
            guarded.setattr(Path, "open", open_readonly)
            yield

    return guard


def retained_snapshot(root):
    return {
        str(path.relative_to(root)): ("link", path.readlink().as_posix()) if path.is_symlink() else path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }


@pytest.mark.parametrize("explicit_seed", [False, True])
def test_cached_sdk_verifies_all_products_and_returns_original_typed_roots(
    cached_sdk, readonly_admission, monkeypatch, explicit_seed
):
    fixture = cached_sdk
    calls = {name: [] for name in ("node_inputs", "compatible_inputs", "verify_node", "verify_dependencies")}

    def observe(name):
        operation = getattr(sdk, name)

        def invoke(*args):
            calls[name].append(args)
            return operation(*args)

        monkeypatch.setattr(sdk, name, invoke)

    for name in calls:
        observe(name)
    before = retained_snapshot(fixture.store.parent)
    kwargs = {"host_seed": fixture.index / "host-seed.json"} if explicit_seed else {}
    with readonly_admission():
        result = sdk.admit_cached_sdk(fixture.ports, fixture.graph, fixture.store, **kwargs)
    assert isinstance(result, MaterializedSDK)
    assert result.sdk == fixture.products["tooling"]
    assert result.llvm == fixture.products["compiler"]
    assert result.sysroot == fixture.products["platform"]
    assert result.cpython_manifest == result.runtime == fixture.products["cpython"]
    assert result.python.runtime_bundle == fixture.products["cpython"].root
    assert result.python.source_root == fixture.products["cpython"].root / "Python-3.13.7"
    assert result.host_tools["python"] == fixture.seed.tools["python"]
    assert {name: result.host_tools[name] for name in fixture.seed.tools} == fixture.seed.tools
    resolver = fixture.products["resolver"]
    assert result.host_tools["uv"] == Tool(
        resolver.root / "uv", resolver.contents["executable"]["sha256"], resolver.path, resolver.sha256
    )
    assert result.compiler() == fixture.products["compiler"].root / "bin/clang"
    expected = [node.name for node in fixture.definition.products]
    for name in ("node_inputs", "verify_node", "verify_dependencies"):
        assert [args[0].name for args in calls[name]] == expected
    assert len(calls["compatible_inputs"]) == 5
    for record, current in calls["compatible_inputs"]:
        assert record["inputs"] == current
    assert retained_snapshot(fixture.store.parent) == before


@pytest.mark.parametrize("product", ["tooling", "compiler", "platform", "cpython", "resolver"])
@pytest.mark.parametrize("damage", ["record", "receipt", "bytes", "missing-artifact", "inputs"])
def test_cached_sdk_missing_or_stale_products_fail_readonly(cached_sdk, readonly_admission, product, damage):
    fixture = cached_sdk
    retained = fixture.products[product]
    record_path = fixture.index / (product + ".json")
    if damage == "record":
        record_path.unlink()
    elif damage == "receipt":
        retained.path.unlink()
    elif damage == "inputs":
        record = sdk.read_json(record_path)
        record["inputs"]["implementation"] = {"stale": "a" * 64}
        record_path.write_text(json.dumps(record))
    else:
        artifact = {
            "tooling": "bin/clang",
            "compiler": "bin/clang",
            "platform": "sysroot/lib/libc.a",
            "cpython": "rootfs/usr/bin/python3.wasm",
            "resolver": "uv",
        }[product]
        if damage == "missing-artifact":
            (retained.root / artifact).unlink()
        else:
            (retained.root / artifact).write_bytes(b"tampered")
    before = retained_snapshot(fixture.store.parent)
    with readonly_admission(), pytest.raises((ValueError, FileNotFoundError)):
        sdk.admit_cached_sdk(fixture.ports, fixture.graph, fixture.store)
    assert retained_snapshot(fixture.store.parent) == before


@pytest.mark.parametrize("product", ["platform", "cpython"])
def test_cached_sdk_rejects_tampered_embedded_dependencies(cached_sdk, readonly_admission, product):
    fixture = cached_sdk
    retained = fixture.products[product]
    contents = sdk.read_json(retained.path)
    if product == "platform":
        contents["identity"]["compiler"] = {}
    else:
        contents["build_profile"]["sysroot"] = {}
    retained.path.write_text(json.dumps(contents))
    record_path = fixture.index / (product + ".json")
    record = sdk.read_json(record_path)
    record["product"]["sha256"] = sdk.file_hash(retained.path)
    record_path.write_text(json.dumps(record))
    before = retained_snapshot(fixture.store.parent)
    with readonly_admission(), pytest.raises(ValueError):
        sdk.admit_cached_sdk(fixture.ports, fixture.graph, fixture.store)
    assert retained_snapshot(fixture.store.parent) == before


@pytest.mark.parametrize("damage", ["host-tool", "python-header", "uv-lock", "schema", "current-policy"])
def test_cached_sdk_rechecks_seed_headers_schema_and_current_policy(cached_sdk, readonly_admission, damage):
    fixture = cached_sdk
    if damage == "host-tool":
        fixture.seed.tools["sh"].path.write_bytes(b"changed seed")
    elif damage == "python-header":
        (fixture.products["cpython"].root / "Python-3.13.7/Include/Python.h").write_bytes(b"changed header")
    elif damage == "current-policy":
        node = next(node for node in fixture.definition.products if node.name == "compiler")
        node.port.recipe["producer_identity"]["version"] = "changed"
    else:
        name = "resolver" if damage == "uv-lock" else "compiler"
        record_path = fixture.index / (name + ".json")
        record = sdk.read_json(record_path)
        if damage == "uv-lock":
            product = fixture.products[name]
            contents = sdk.read_json(product.path)
            contents["build"]["locked"] = False
            product.path.write_text(json.dumps(contents))
            record["product"]["sha256"] = sdk.file_hash(product.path)
        else:
            record["schema_version"] = True
        record_path.write_text(json.dumps(record))
    before = retained_snapshot(fixture.store.parent)
    with readonly_admission(), pytest.raises(ValueError):
        sdk.admit_cached_sdk(fixture.ports, fixture.graph, fixture.store)
    assert retained_snapshot(fixture.store.parent) == before


def test_cached_sdk_missing_store_creates_nothing(tmp_path, readonly_admission):
    ports = Path(__file__).resolve().parents[2]
    store = tmp_path / "absent"
    with readonly_admission(), pytest.raises(FileNotFoundError):
        sdk.admit_cached_sdk(ports, Graph((), ()), store)
    assert not store.exists()


def test_sdk_admission_module_is_outside_all_producer_implementation_closures():
    ports = Path(__file__).resolve().parents[2]
    definition = sdk.definition(ports, Graph((), ()))
    assert {node.name for node in definition.products} == {"tooling", "compiler", "platform", "cpython", "resolver"}
    for node in definition.products:
        assert "_support/sdk.py" not in sdk.implementation(node.port)
        assert all(
            (node.recipe_path.parent / item["file"]).resolve() != Path(sdk.__file__).resolve()
            for item in node.recipe["build_scripts"]
        )
