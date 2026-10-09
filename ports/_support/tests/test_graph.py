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
