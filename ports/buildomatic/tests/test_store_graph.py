"""Exercise metadata planning, dependency readiness and retry identities offline."""

from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic import LocalStore, extract_tree
from ports.buildomatic.artifacts import register, resolve
from ports.buildomatic.backends.iris import CoordinatorService
from ports.buildomatic.store_graph import prepare


class CatalogBackend:
    def __init__(self, store):
        self.store = store
        self.uploads = []

    def resolve_artifact(self, key):
        return resolve(self.store, key)

    def upload_bundles(self, source, bundles):
        self.uploads.extend(bundles)


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
    register(backend.store, "port/" + keys[base], {"artifact": {"id": "base"}})
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
    reference = {"artifact": {"id": "stored-product"}}
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
