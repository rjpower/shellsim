"""Exercise dependency selection before any fetching, compilation or staging."""

import json
from pathlib import Path

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


def test_guest_host_wheel_error_identifies_complete_dependency_chain(tmp_path):
    root = tmp_path / "ports"
    for name, adapter, dependency_name in [
        ("consumer", "pure-wheel", "cppy"),
        ("cppy", "pure-wheel", "setuptools"),
        ("setuptools", "host-wheel", None),
    ]:
        path = root / "python" / name / "recipe.json"
        path.parent.mkdir(parents=True)
        recipe = {"name": name, "version": "1", "build_system": adapter}
        if adapter == "host-wheel":
            recipe["role"] = "host-tool"
        if dependency_name is not None:
            recipe["runtime_dependencies"] = [{"port": "python/" + dependency_name, "version": "1"}]
        path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError) as failure:
        plan(root, ["python/consumer"])
    assert str(failure.value) == (
        "host-only provider in guest dependency chain: python/consumer/recipe.json -> "
        "python/cppy/recipe.json -> python/setuptools/recipe.json"
    )


@pytest.fixture
def sdk_recipes(recipes):
    root, write = recipes
    compiler = write("compiler", variant="host.json", profile=None)
    compiler.write_text(json.dumps({"name": "compiler", "version": "1", "role": "host-tool"}))
    platform = write("platform")
    platform.write_text(json.dumps({"name": "platform", "version": "1", "role": "target-platform"}))
    write("base")
    write("base", variant="shared.json")
    consumer = write("consumer", dependencies=[dependency("base")], profile=None)
    recipe = json.loads(consumer.read_text())
    recipe.pop("target_profile")
    recipe["sdk"] = "test"
    consumer.write_text(json.dumps(recipe))
    profile = root / "sdks/test.json"
    profile.parent.mkdir()
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "host": "linux-x86_64",
                "products": {},
                "target": "wasm32-wasip1-threads",
                "target_profile": "test-target",
                "abi": "test-abi",
                "build_dependencies": [dependency("compiler", recipe="native/compiler/host.json")],
                "platform_dependencies": [dependency("platform")],
            }
        )
    )
    return root, write, consumer, profile


def test_sdk_resolves_pinned_native_closure_and_explicit_overrides(sdk_recipes):
    root, write, consumer, _ = sdk_recipes
    graph = plan(root, ["native/consumer"], target_profile="test-target")
    assert [(port.reference, port.role) for port in graph.ports] == [
        ("native/compiler/host.json", "host-tool"),
        ("native/base/recipe.json", "target-library"),
        ("native/platform/recipe.json", "target-platform"),
        ("native/consumer/recipe.json", "target-library"),
    ]
    resolved = graph.ports[-1]
    assert {(edge.recipe, edge.kind) for edge in resolved.dependencies} == {
        ("native/compiler/host.json", "build"),
        ("native/base/recipe.json", "target"),
        ("native/platform/recipe.json", "platform"),
    }
    assert resolved.recipe["target_dependencies"] == [
        {"port": "native/base", "version": "1", "recipe": "native/base/recipe.json"}
    ]
    write("base", variant="alternate.json")
    authored = json.loads(consumer.read_text())
    authored["target_dependencies"][0]["recipe"] = "native/base/alternate.json"
    consumer.write_text(json.dumps(authored))
    overridden = plan(root, ["native/consumer"]).ports[-1]
    assert overridden.dependencies[1].recipe == "native/base/alternate.json"
    with pytest.raises(ValueError, match="conflicting recipe"):
        plan(root, ["native/consumer", "native/base/shared.json"])


def test_sdk_cannot_relax_dependency_version_or_cohort(sdk_recipes):
    root, _, consumer, profile = sdk_recipes
    with pytest.raises(ValueError, match="target profile differs"):
        plan(root, ["native/consumer"], target_profile="other-target")
    authored = json.loads(consumer.read_text())
    authored["target_dependencies"][0]["version"] = "2"
    consumer.write_text(json.dumps(authored))
    with pytest.raises(ValueError, match="name or version differs"):
        plan(root, ["native/consumer"])
    profile.unlink()
    with pytest.raises(ValueError, match="dependency chain"):
        plan(root, ["native/consumer"])


@pytest.mark.parametrize(
    "failure", ["scalar", "compiler", "host", "pin", "unknown", "schema", "variant", "oversize", "link"]
)
def test_invalid_sdks_fail_before_graph_admission(sdk_recipes, failure):
    root, _, consumer, profile_path = sdk_recipes
    recipe = json.loads(consumer.read_text())
    profile = json.loads(profile_path.read_text())
    if failure == "scalar":
        recipe["abi"] = "other-abi"
    elif failure == "compiler":
        recipe["build_dependencies"] = [dependency("compiler", recipe="native/compiler/host.json")]
    elif failure == "host":
        recipe["role"] = "host-tool"
    elif failure == "pin":
        profile["build_dependencies"][0].pop("version")
    elif failure == "unknown":
        profile["inherits"] = "another-profile"
    elif failure == "schema":
        profile["schema_version"] = True
    elif failure == "variant":
        profile["dependency_recipes"] = {"native/base": "../escape.json"}
    consumer.write_text(json.dumps(recipe))
    profile_path.write_text(json.dumps(profile))
    if failure == "oversize":
        profile_path.write_bytes(b" " * (1024 * 1024 + 1))
    elif failure == "link":
        saved = profile_path.with_suffix(".saved")
        profile_path.rename(saved)
        profile_path.symlink_to(saved)
    with pytest.raises(ValueError):
        plan(root, ["native/consumer"])


@pytest.mark.parametrize("name", ["../escape", "test.json", "Test", "", None, ["test"]])
def test_sdk_names_cannot_escape_the_named_registry(sdk_recipes, name):
    root, _, consumer, _ = sdk_recipes
    recipe = json.loads(consumer.read_text())
    recipe["sdk"] = name
    consumer.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        plan(root, ["native/consumer"])


def test_direct_and_transitive_ports_select_one_canonical_provider():
    root = Path(__file__).resolve().parents[2]
    graph = plan(root, ["native/zlib", "native/freetype", "python/pillow"])
    ports = {port.reference: port for port in graph.ports}
    for reference in ("native/freetype/recipe.json", "python/pillow/recipe.json"):
        edges = ports[reference].dependencies
        assert {edge.recipe for edge in edges if edge.kind == "build"} >= {"toolchain/llvm/recipe.json:host"}
        assert [edge.recipe for edge in edges if edge.kind == "platform"] == [
            "toolchain/wasi_threads/recipe.json:platform"
        ]
        zlib = next(edge for edge in edges if edge.port == "native/zlib")
        assert zlib.recipe == "native/zlib/recipe.json"
        assert ports[zlib.recipe].version == zlib.version
    assert len([port for port in graph.ports if port.name == "zlib"]) == 1


def test_every_production_selection_resolves_its_exact_dependency_graph():
    root = Path(__file__).resolve().parents[2]
    for category in ("native", "python", "toolchain"):
        for path in sorted((root / category).glob("*/recipe.json")):
            document = json.loads(path.read_text())
            reference = path.parent.relative_to(root).as_posix()
            variants = document.get("variants", {})
            for selection in tuple(reference + ":" + name for name in variants) if variants else (reference,):
                graph = plan(root, [selection])
                providers = {port.reference: port for port in graph.ports}
                for port in graph.ports:
                    for edge in port.dependencies:
                        assert providers[edge.recipe].version == edge.version


def test_default_variant_and_explicit_selection_share_node_without_importing_builder(tmp_path):
    directory = tmp_path / "native/example"
    directory.mkdir(parents=True)
    (directory / "build.py").write_text("raise AssertionError('planning imported build code')\n")
    (directory / "recipe.json").write_text(
        json.dumps(
            {
                "name": "example",
                "version": "1",
                "role": "host-tool",
                "default_variant": "host",
                "variants": {"host": {}, "other": {"name": "other"}},
            }
        )
    )
    graph = plan(tmp_path, ["native/example", "native/example:host"])
    assert len(graph.roots) == len(graph.ports) == 1
    assert graph.ports[0].variant == "host"
    assert plan(tmp_path, ["native/example:other"]).ports[0].digest != graph.ports[0].digest
