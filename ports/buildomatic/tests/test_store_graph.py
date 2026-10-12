"""Retain qualified SDK compatibility and independent canonical storage."""

from ports.buildomatic import LocalStore
from ports.buildomatic.artifacts import register
from ports.buildomatic.backends.iris import CoordinatorService


def test_sdk_interface_changes_keep_products_but_producer_changes_do_not():
    from copy import deepcopy

    from ports.buildomatic.sdk_artifacts import sdk_inputs_match

    current = {
        "definition": {"target": "wasm32-wasip1-threads"},
        "producers": {
            "compiler": {
                "policy": {"version": "23"},
                "implementation": {"builder": "dispatcher", "toolchain/llvm/compiler.py": "compiler-code"},
            }
        },
    }
    recorded = deepcopy(current)
    recorded["producers"]["compiler"]["implementation"]["_support/build_context.py"] = "old-interface"
    recorded["producers"]["compiler"]["implementation"]["builder"] = "old-dispatcher"
    assert sdk_inputs_match(recorded, current)
    for field, value in [
        ("policy", {"version": "24"}),
        ("implementation", {"toolchain/llvm/compiler.py": "changed-compiler"}),
    ]:
        changed = deepcopy(current)
        changed["producers"]["compiler"][field] = value
        assert not sdk_inputs_match(recorded, changed)


def test_artifact_catalog_is_independent_of_service_journals(tmp_path):
    canonical = LocalStore(tmp_path / "canonical")
    register(canonical, "sdk/example", {"identity": "sdk"})
    service = CoordinatorService(LocalStore(tmp_path / "service"), "test", {}, artifact_store=canonical)
    assert service.resolve_artifact("sdk/example") == {"identity": "sdk"}
