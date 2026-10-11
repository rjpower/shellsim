"""Exercise metadata planning, dependency readiness and retry identities offline."""

from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic import LocalStore, extract_tree
from ports.buildomatic.artifacts import register, resolve
from ports.buildomatic.backends.iris import CoordinatorService
from ports.buildomatic.store_graph import prepare


def test_sdk_interface_changes_keep_products_but_producer_changes_do_not():
    from copy import deepcopy

    from ports.buildomatic.sdk_artifacts import sdk_inputs_match

    current = {
        "definition": {"target": "wasm32-wasip1-threads"},
        "producers": {"compiler": {"policy": {"version": "23"}, "implementation": {"builder": "compiler-code"}}},
    }
    recorded = deepcopy(current)
    recorded["producers"]["compiler"]["implementation"]["_support/build_context.py"] = "old-interface"
    assert sdk_inputs_match(recorded, current)
    for field, value in [("policy", {"version": "24"}), ("implementation", {"builder": "changed-compiler"})]:
        changed = deepcopy(current)
        changed["producers"]["compiler"][field] = value
        assert not sdk_inputs_match(recorded, changed)


class CatalogBackend:
    def __init__(self, store):
        self.store = store
        self.uploads = []

    def resolve_artifact(self, key):
        return resolve(self.store, key)

    def upload_bundles(self, source, bundles):
        self.uploads.extend(bundles)


def cached_product(reference, key):
    return {
        "reference": reference,
        "artifact": {"id": reference},
        "delivery": {"reference": reference, "build_key": key, "wheels": [], "native": []},
    }


def test_transitive_inputs_and_partial_retry_keep_product_keys(tmp_path):
    ports = tmp_path / "ports"
    for name, dependencies in (("base", []), ("provider", ["base"]), ("consumer", ["provider"])):
        _wheel_port(ports, tmp_path / "sources", name, dependencies)
    local = LocalStore(tmp_path / "transport")
    backend = CatalogBackend(LocalStore(tmp_path / "canonical"))
    args = ports, ["python/consumer"], None, local, backend, "s3://bucket/cache", tmp_path
    graph, first, results, keys = prepare(*args, 16, 2)
    assert not results
    assert len(first.actions) == 3
    consumer = first.actions[-1]
    assert set(consumer.dependencies) == {action.id for action in first.actions[:2]}
    recipe = tmp_path / "recipe"
    extract_tree(consumer.inputs[1].bundle, local, recipe)
    import json

    specification = json.loads((recipe / "node.json").read_text())
    assert set(specification["predecessors"]) == {port.reference for port in graph.ports[:2]}

    base = graph.ports[0].reference
    register(backend.store, "port/" + keys[base], cached_product(base, keys[base]))
    _, partial, results, partial_keys = prepare(*args, 32, 4)
    assert partial_keys == keys
    assert base in results
    assert len(partial.actions) == 2
    assert partial.idempotency_key != first.idempotency_key
    _, fresh, _, _ = prepare(*args, 32, 4, build_key="retry")
    assert fresh.idempotency_key != partial.idempotency_key


def test_empty_client_hit_uses_only_catalog_metadata(tmp_path, monkeypatch):
    ports = tmp_path / "ports"
    _wheel_port(ports, tmp_path / "sources", "example", [])
    backend = CatalogBackend(LocalStore(tmp_path / "canonical"))
    local = LocalStore(tmp_path / "first-client")
    args = ports, ["python/example"], None, local, backend, "s3://bucket/cache", tmp_path
    _, _, _, keys = prepare(*args, 16, 1)
    reference = cached_product("python/example/recipe.json", next(iter(keys.values())))
    register(backend.store, "port/" + next(iter(keys.values())), reference)
    backend.uploads.clear()

    def reject_product_read(*args):
        raise AssertionError("planning must not fetch a product")

    monkeypatch.setattr(backend.store, "get_blob", reject_product_read)
    args = (
        ports,
        ["python/example"],
        None,
        LocalStore(tmp_path / "empty-client"),
        backend,
        "s3://bucket/cache",
        tmp_path,
    )
    _, request, results, _ = prepare(*args, 128, 8)
    assert request is None
    assert list(results.values()) == [reference]
    assert not backend.uploads


def test_artifact_catalog_is_independent_of_service_journals(tmp_path):
    canonical = LocalStore(tmp_path / "canonical")
    register(canonical, "sdk/example", {"identity": "sdk"})
    service = CoordinatorService(LocalStore(tmp_path / "service"), "test", {}, artifact_store=canonical)
    assert service.resolve_artifact("sdk/example") == {"identity": "sdk"}


def test_saved_roots_keep_build_only_wheels_out_of_guest_release(tmp_path, monkeypatch):
    import json
    from contextlib import nullcontext

    from ports._support.graph import guest_graph, plan
    from ports.buildomatic.backends import iris
    from ports.buildomatic.store_graph import run

    ports = tmp_path / "ports"
    for name, dependencies in (("generator", []), ("runtime", []), ("application", ["runtime"])):
        _wheel_port(ports, tmp_path / "sources", name, dependencies)
    recipe = ports / "python/application/recipe.json"
    data = json.loads(recipe.read_text())
    data["build_dependencies"] = [{"port": "python/generator", "version": "1"}]
    recipe.write_text(json.dumps(data))
    backend = CatalogBackend(LocalStore(tmp_path / "canonical"))
    scratch = tmp_path / "client"
    scratch.mkdir()
    _, _, _, keys = prepare(
        ports,
        ["python/application"],
        None,
        LocalStore(scratch / "transport"),
        backend,
        "s3://bucket/cache",
        scratch,
        32,
        8,
    )
    for reference, key in keys.items():
        register(backend.store, "port/" + key, cached_product(reference, key))
    descriptor = tmp_path / "service.json"
    descriptor.write_text(
        json.dumps(
            {
                "job_id": "test",
                "cluster_name": "test",
                "controller_url": "https://controller.invalid",
                "workspace": str(tmp_path),
                "prefix": "s3://bucket/cache",
            }
        )
    )
    monkeypatch.setattr(iris, "backend_context_factory", lambda **kwargs: lambda: nullcontext(backend))
    run(ports, ["python/application"], "fixture", descriptor, scratch, jobs=32, max_workers=8)
    saved = json.loads((scratch / "results.json").read_text())
    assert len(saved["products"]) == 3
    graph = plan(ports, saved["roots"])
    assert {port.name for port in guest_graph(graph).ports} == {"application", "runtime"}


def test_sdk_fetch_selection_follows_transitive_port_inputs():
    from pathlib import Path

    from ports._support.graph import plan
    from ports.buildomatic.store_graph import consumer_sdk

    ports = Path(__file__).resolve().parents[2]
    manifest = {
        "products": {role: {"role": role} for role in ("compiler", "platform", "tooling", "cpython", "resolver")}
    }
    native = consumer_sdk(manifest, plan(ports, ["native/zlib"]))
    assert set(native["products"]) == {"compiler", "platform", "tooling"}
    imageio = consumer_sdk(manifest, plan(ports, ["python/imageio"]))
    assert set(imageio["products"]) == {"compiler", "platform", "tooling", "cpython"}
    assert consumer_sdk(manifest, plan(ports, ["python/pytest-json-report"])) is None
    assert set(manifest["products"]) == {"compiler", "platform", "tooling", "cpython", "resolver"}


def test_pure_product_key_does_not_depend_on_incidental_sdk(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from ports._support import sdk as policies

    ports = tmp_path / "ports"
    _wheel_port(ports, tmp_path / "sources", "example", [])
    monkeypatch.setattr(policies, "definition", lambda *args: SimpleNamespace(products=()))
    local = LocalStore(tmp_path / "transport")
    backend = CatalogBackend(LocalStore(tmp_path / "canonical"))
    args = local, backend, "s3://bucket/cache", tmp_path, 32, 8
    _, standalone, _, first = prepare(ports, ["python/example"], None, *args)
    manifest = {"identity": "incidental-sdk", "dynamic_abi": "fixture", "products": {}}
    _, mixed, _, second = prepare(ports, ["python/example"], manifest, *args)
    assert first == second
    node = tmp_path / "node"
    extract_tree(mixed.actions[0].inputs[1].bundle, local, node)
    assert json.loads((node / "node.json").read_text())["sdk"] is None
    assert standalone.actions[0].id == mixed.actions[0].id


def test_legacy_product_backfills_assets_without_sdk_or_dependencies(tmp_path):
    import json

    ports = tmp_path / "ports"
    _wheel_port(ports, tmp_path / "sources", "example", [])
    local = LocalStore(tmp_path / "transport")
    backend = CatalogBackend(LocalStore(tmp_path / "canonical"))
    args = ports, ["python/example"], None, local, backend, "s3://bucket/cache", tmp_path
    _, _, _, keys = prepare(*args, 32, 8)
    reference = next(iter(keys))
    existing = {"reference": reference, "artifact": {"id": "legacy-product"}}
    register(backend.store, "port/" + keys[reference], existing)
    _, request, _, retry_keys = prepare(*args, 128, 8)
    assert retry_keys == keys
    assert len(request.actions) == 1
    action = request.actions[0]
    assert action.id.startswith("delivery-") and action.dependencies == ()
    node = tmp_path / "legacy-node"
    extract_tree(action.inputs[1].bundle, local, node)
    specification = json.loads((node / "node.json").read_text())
    assert specification["sdk"] is None and specification["existing"] == existing
