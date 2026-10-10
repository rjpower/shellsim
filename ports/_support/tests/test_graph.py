"""Exercise dependency selection before any fetching, compilation or staging."""

import json

import pytest

from ports._support.graph import plan


@pytest.fixture
def recipes(tmp_path):
    def write(name, *, dependencies=(), version="1", variant="recipe.json", profile="test-target"):
        path = tmp_path / "native" / name / variant
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": name,
                    "version": version,
                    "target_profile": profile,
                    "target_dependencies": list(dependencies),
                }
            )
        )
        return path

    return tmp_path, write


def dependency(name, *, version="1", recipe=None):
    value = {"port": "native/" + name, "version": version}
    if recipe is not None:
        value["recipe"] = recipe
    return value


def test_deep_diamond_builds_each_provider_once_before_consumers(recipes):
    root, write = recipes
    write("base")
    write("middle", dependencies=[dependency("base")])
    write("left", dependencies=[dependency("middle")])
    write("right", dependencies=[dependency("middle")])
    write("top", dependencies=[dependency("left"), dependency("right")])
    graph = plan(root, ["native/top", "native/right"], target_profile="test-target")
    assert [port.name for port in graph.ports] == ["base", "middle", "left", "right", "top"]


def test_explicit_variant_never_falls_back_to_default(recipes):
    root, write = recipes
    write("base", version="9")
    write("base", variant="shared-recipe.json")
    write("consumer", dependencies=[dependency("base", recipe="native/base/shared-recipe.json")])
    graph = plan(root, ["native/consumer"])
    assert graph.ports[0].reference == "native/base/shared-recipe.json"
    (root / "native/base/shared-recipe.json").unlink()
    with pytest.raises(ValueError, match="dependency chain"):
        plan(root, ["native/consumer"])


@pytest.mark.parametrize("failure", ["cycle", "version", "profile", "variant", "duplicate", "name"])
def test_inconsistent_graph_is_rejected(recipes, failure):
    root, write = recipes
    write("base")
    write("child", dependencies=[dependency("base")])
    write("top", dependencies=[dependency("child")])
    if failure == "cycle":
        write("base", dependencies=[dependency("top")])
    elif failure == "version":
        write("base", version="2")
    elif failure == "profile":
        write("base", profile="other-target")
    elif failure == "variant":
        write("base", variant="shared-recipe.json")
        write("top", dependencies=[dependency("child"), dependency("base", recipe="native/base/shared-recipe.json")])
    elif failure == "duplicate":
        write("child", dependencies=[dependency("base"), dependency("base")])
    else:
        write("child", dependencies=[dependency("wrong", recipe="native/base/recipe.json")])
    with pytest.raises(ValueError):
        plan(root, ["native/top"], target_profile="test-target")


def test_changed_recipe_changes_input_identity(recipes):
    root, write = recipes
    write("base")
    before = plan(root, ["native/base"]).ports[0].digest
    write("base", version="2")
    assert plan(root, ["native/base"]).ports[0].digest != before


def test_conflicting_root_and_dependency_variants_are_rejected(recipes):
    root, write = recipes
    write("base")
    write("base", variant="shared-recipe.json")
    write("consumer", dependencies=[dependency("base", recipe="native/base/shared-recipe.json")])
    with pytest.raises(ValueError, match="conflicting recipe"):
        plan(root, ["native/base", "native/consumer"])


@pytest.mark.parametrize("reference", ["../escape", "/absolute", "native//base", "native/./base"])
def test_recipe_references_stay_inside_ports_tree(recipes, reference):
    root, _ = recipes
    with pytest.raises(ValueError):
        plan(root, [reference])


def test_linked_recipe_directory_is_rejected(recipes):
    root, write = recipes
    write("base")
    (root / "native/alias").symlink_to(root / "native/base", target_is_directory=True)
    with pytest.raises(ValueError):
        plan(root, ["native/alias"])


def test_excessive_dependency_depth_is_rejected(recipes):
    root, write = recipes
    for index in range(65):
        write(str(index), dependencies=[dependency(str(index + 1))] if index < 64 else [])
    with pytest.raises(ValueError, match="size limit"):
        plan(root, ["native/0"])


def test_oversized_recipe_is_rejected_before_json_parse(recipes):
    root, write = recipes
    path = write("base")
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError) as caught:
        plan(root, ["native/base"])
    assert "size limit" in str(caught.value.__cause__)


def test_host_and_guest_variants_have_distinct_dependency_roles(recipes):
    root, write = recipes
    host = write("compiler", variant="host.json", profile=None)
    host.write_text(
        json.dumps({"name": "compiler", "version": "1", "role": "host-tool", "target_profile": "build-machine"})
    )
    guest = write("compiler")
    guest.write_text(
        json.dumps(
            {
                "name": "compiler",
                "version": "1",
                "role": "guest-tool",
                "build_dependencies": [dependency("compiler", recipe="native/compiler/host.json")],
            }
        )
    )
    graph = plan(root, ["native/compiler"], target_profile="test-target")
    assert [port.role for port in graph.ports] == ["host-tool", "guest-tool"]
    assert graph.ports[1].dependencies[0].kind == "build"


@pytest.mark.parametrize(
    "field,role",
    [
        ("build_dependencies", "target-library"),
        ("target_dependencies", "host-tool"),
        ("runtime_dependencies", "host-tool"),
        ("platform_dependencies", "target-library"),
    ],
)
def test_wrong_dependency_role_is_rejected(recipes, field, role):
    root, write = recipes
    provider = write("provider")
    provider.write_text(json.dumps({"name": "provider", "version": "1", "role": role}))
    consumer = write("consumer")
    consumer.write_text(json.dumps({"name": "consumer", "version": "1", field: [dependency("provider")]}))
    with pytest.raises(ValueError, match="role differs"):
        plan(root, ["native/consumer"])


def test_runtime_and_platform_edges_remain_distinct_from_link_inputs(recipes):
    root, write = recipes
    platform = write("platform")
    platform.write_text(json.dumps({"name": "platform", "version": "1", "role": "target-platform"}))
    write("runtime")
    write("library")
    consumer = write("consumer")
    consumer.write_text(
        json.dumps(
            {
                "name": "consumer",
                "version": "1",
                "target_dependencies": [dependency("library")],
                "runtime_dependencies": [dependency("runtime")],
                "platform_dependencies": [dependency("platform")],
            }
        )
    )
    graph = plan(root, ["native/consumer"])
    assert {(edge.port, edge.kind) for edge in graph.ports[-1].dependencies} == {
        ("native/library", "target"),
        ("native/runtime", "runtime"),
        ("native/platform", "platform"),
    }
