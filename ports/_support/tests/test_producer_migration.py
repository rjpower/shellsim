"""Frozen prior identities admit real inventories only for reviewed producer code."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ports._support import producer_migration, producer_policy
from ports._support.graph import Port
from ports._support.sdk_products import Receipt, file_hash, json_hash
from ports.api import implementation


@pytest.fixture
def migrated_product(tmp_path, monkeypatch):
    directory = tmp_path / "ports/native/example"
    directory.mkdir(parents=True)
    (directory / "build.py").write_text("def build(ctx): return ctx.result\n")
    helper = directory / "helper.py"
    helper.write_text("FLAGS = ('-O2',)\n")
    metadata = {
        "name": "example",
        "version": "1",
        "build_system": "llvm-host",
        "source": {"sha256": "a" * 64},
        "patches": [],
        "protocol": {},
        "build_limits": {},
        "main_tls_protocol": {},
        "helpers": ["native/example/helper.py"],
    }
    port = Port("native/example/recipe.json", directory, "example", "1", "", (), metadata)
    node = SimpleNamespace(name="compiler", adapter="llvm-host", port=port)
    original = {**producer_policy.policy(metadata), "build_scripts": [{"file": "build.py", "sha256": "b" * 64}]}
    root = tmp_path / "product"
    root.mkdir()
    (root / "compiler").write_bytes(b"verified compiler fixture")
    contents = {"identity": {"recipe": original}, "artifacts": {"compiler": file_hash(root / "compiler")}}
    path = root / "manifest.json"
    path.write_text(json.dumps(contents))
    product = Receipt(root, path, file_hash(path), contents)
    registry = {
        "products": {
            "compiler": {
                "source_policy_sha256": ["c" * 64],
                "receipt_policy_sha256": [json_hash(original)],
                "implementation_sha256": json_hash(implementation(port)),
            }
        }
    }
    read = producer_migration.read_json
    monkeypatch.setattr(
        producer_migration,
        "read_json",
        lambda path: registry if path.name == "producer-migration-v1.json" else read(path),
    )
    monkeypatch.setattr(producer_policy, "metadata", lambda *_args: (directory, metadata))
    return node, product, helper, registry


def test_original_receipt_and_bytes_remain_unchanged(migrated_product):
    node, product, _, _ = migrated_product
    before = product.path.read_bytes(), product.sha256
    producer_migration.admit_migration(node, "c" * 64, product)
    assert (product.path.read_bytes(), file_hash(product.path)) == before


@pytest.mark.parametrize(
    "change", ["builder", "helper", "prior-policy", "prior-implementation", "inventory", "source", "unknown-field"]
)
def test_unreviewed_equivalence_is_rejected(migrated_product, change):
    node, product, helper, registry = migrated_product
    original = product.contents["identity"]["recipe"]
    if change == "builder":
        (node.port.directory / "build.py").write_text("def build(ctx): return None\n")
    elif change == "helper":
        helper.write_text("FLAGS = ('-O3',)\n")
    elif change == "inventory":
        (product.root / "compiler").write_bytes(b"corrupt compiler")
    elif change in {"prior-implementation", "source", "unknown-field"}:
        changed = dict(original)
        if change == "prior-implementation":
            changed["build_scripts"] = [{"file": "build.py", "sha256": "d" * 64}]
        elif change == "source":
            changed["source"] = {"sha256": "e" * 64}
        else:
            changed["unreviewed_flags"] = ["-O3"]
        product = replace(product, contents={**product.contents, "identity": {"recipe": changed}})
        if change != "prior-implementation":
            registry["products"]["compiler"]["receipt_policy_sha256"] = [json_hash(changed)]
    with pytest.raises(ValueError):
        producer_migration.admit_migration(node, "d" * 64 if change == "prior-policy" else "c" * 64, product)
