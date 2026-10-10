"""Plan a bounded ports dependency graph before fetching or building anything.

Recipe references are relative to the ports tree. A directory selects recipe.json;
other variants can be named explicitly or selected by a recipe-owned SDK.
SDK defaults expand into explicit inputs before validation. The build driver consumes
dependency-first nodes and records their digests alongside artifact identities.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from packaging.version import Version

_MAX_RECIPE_BYTES = 1024 * 1024
_MAX_NODES = 512
_MAX_DEPTH = 64


@dataclass(frozen=True)
class Dependency:
    """One exact recipe selection under a consumer's logical dependency name."""

    port: str
    version: str
    recipe: str
    kind: str = "target"


@dataclass(frozen=True)
class SDKSelection:
    """The selected SDK document and the digest of its consumer defaults."""

    reference: str
    sha256: str


@dataclass(frozen=True)
class Port:
    """A parsed recipe and the identities needed to admit its build inputs."""

    reference: str
    directory: Path
    name: str
    version: str
    digest: str
    dependencies: tuple[Dependency, ...]
    recipe: dict[str, Any]
    sdk_selection: SDKSelection | None = None

    @property
    def role(self) -> str:
        return self.recipe.get("role", "target-library")


@dataclass(frozen=True)
class Graph:
    """Selected roots and unique ports ordered with dependencies first."""

    roots: tuple[str, ...]
    ports: tuple[Port, ...]


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\\" in value or "\0" in value:
        raise ValueError("port reference must be a relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts) or path.as_posix() != value:
        raise ValueError("port reference must be a canonical relative path")
    return value


def _reference(value: object) -> str:
    name = _relative(value)
    return name if name.endswith(".json") else name + "/recipe.json"


def _document(root: Path, reference: str) -> tuple[Path, bytes, dict[str, Any]]:
    """Read a bounded, unlinked JSON document inside the admitted ports tree."""
    path = root
    for part in PurePosixPath(reference).parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"linked recipe path is unsupported: {reference}")
    with path.open("rb") as stream:
        data = stream.read(_MAX_RECIPE_BYTES + 1)
    if len(data) > _MAX_RECIPE_BYTES:
        raise ValueError(f"recipe exceeds its size limit: {reference}")
    recipe = json.loads(data)
    if not isinstance(recipe, dict):
        raise ValueError(f"recipe must be an object: {reference}")
    return path, data, recipe


def _expand_sdk(root: Path, recipe: dict[str, Any], default_sdk: str) -> tuple[dict[str, Any], SDKSelection | None]:
    """Resolve recipe-owned defaults without changing pins or explicit variants.

    SDK defaults join the authoring recipe in its identity. Product recipes and
    inventories have separate identities, so resolver changes do not recompile
    native consumers.
    """
    if "build_profile" in recipe:
        raise ValueError("build_profile was replaced by sdk selection")
    adapter = recipe.get("build", {}).get("adapter")
    explicit = "sdk" in recipe
    if recipe.get("role") in {"host-tool", "target-platform"} or adapter in {"pure-wheel", "host-wheel"}:
        if explicit:
            raise ValueError("SDK selection applies only to target consumers")
        return recipe, None
    if not explicit and adapter is None:
        return recipe, None
    name = recipe.get("sdk", "default")
    if name == "default":
        name = default_sdk
    if name == "default":
        _, _, alias = _document(root, "sdks/default.json")
        if set(alias) != {"schema_version", "sdk"} or alias["schema_version"] != 1:
            raise ValueError("invalid default SDK alias")
        name = alias["sdk"]
    if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) is None:
        raise ValueError("SDK must be a simple lowercase name")
    if recipe.get("role") in {"host-tool", "target-platform"}:
        raise ValueError("SDKs apply only to target consumers")
    _, data, profile = _document(root, f"sdks/{name}.json")
    fields = {
        "schema_version",
        "target",
        "target_profile",
        "abi",
        "build_dependencies",
        "platform_dependencies",
        "dependency_recipes",
        "host",
        "products",
    }
    if set(profile) != fields or type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("unsupported SDK schema")
    _dependencies(recipe, name)
    resolved = dict(recipe)
    for field in ("target", "target_profile", "abi"):
        value = profile[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"SDK needs {field}")
        if field in recipe and recipe[field] != value:
            raise ValueError(f"recipe conflicts with SDK {field}")
        resolved[field] = value
    # Validate profile edges before merging, so malformed defaults cannot be hidden.
    _dependencies(profile, f"sdks/{name}.json")
    for field in ("build_dependencies", "platform_dependencies"):
        declarations = recipe.get(field, [])
        if not isinstance(declarations, list) or len(declarations) > _MAX_NODES:
            raise ValueError("invalid recipe dependency declarations")
        owned = {item["port"] for item in profile[field]}
        if any(isinstance(item, dict) and item.get("port") in owned for item in declarations):
            raise ValueError(f"recipe redeclares SDK {field}")
        resolved[field] = [*profile[field], *declarations]
    variants = profile["dependency_recipes"]
    if not isinstance(variants, dict) or len(variants) > _MAX_NODES:
        raise ValueError("invalid SDK dependency variants")
    variants = {_relative(port): _reference(reference) for port, reference in variants.items()}
    for field in ("build_dependencies", "target_dependencies", "runtime_dependencies", "platform_dependencies"):
        declarations = resolved.get(field, [])
        if not isinstance(declarations, list) or len(declarations) > _MAX_NODES:
            raise ValueError("invalid recipe dependency declarations")
        resolved[field] = [
            {**item, "recipe": _reference(item.get("recipe", variants.get(item["port"], item["port"])))}
            for item in declarations
        ]
    return resolved, SDKSelection(
        f"sdks/{name}.json",
        hashlib.sha256(
            json.dumps({key: profile[key] for key in fields - {"products", "host"}}, sort_keys=True).encode()
        ).hexdigest(),
    )


def _dependencies(recipe: dict[str, Any], reference: str) -> tuple[Dependency, ...]:
    """Validate exact role-specific edges after SDK expansion."""
    dependencies = []
    for field, kind in (
        ("build_dependencies", "build"),
        ("target_dependencies", "target"),
        ("runtime_dependencies", "runtime"),
        ("platform_dependencies", "platform"),
    ):
        declarations = recipe.get(field, [])
        if not isinstance(declarations, list) or len(declarations) > _MAX_NODES:
            raise ValueError(f"invalid dependency declarations: {reference}")
        names = set()
        for declaration in declarations:
            if not isinstance(declaration, dict):
                raise ValueError(f"dependency must be an object: {reference}")
            port = _relative(declaration.get("port"))
            version_pin = declaration.get("version")
            if not isinstance(version_pin, str) or not version_pin:
                raise ValueError(f"dependency needs an exact build version: {reference} -> {port}")
            Version(version_pin)
            if port in names:
                raise ValueError(f"duplicate dependency: {reference} -> {port}")
            names.add(port)
            dependencies.append(Dependency(port, version_pin, _reference(declaration.get("recipe", port)), kind))
    return tuple(dependencies)


def _read(root: Path, reference: str, default_sdk: str) -> Port:
    path, data, recipe = _document(root, reference)
    recipe, profile = _expand_sdk(root, recipe, default_sdk)
    name, version = recipe.get("name"), recipe.get("version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ValueError(f"recipe must declare a name and version: {reference}")
    if PurePosixPath(name).name != name or name in {".", ".."}:
        raise ValueError(f"recipe name must be a package name: {reference}")
    Version(version)
    role = recipe.get("role", "target-library")
    if role not in {"host-tool", "target-library", "guest-tool", "target-platform"}:
        raise ValueError(f"unsupported port role: {reference}")
    dependencies = _dependencies(recipe, reference)
    digest = hashlib.sha256(data).hexdigest()
    if profile is not None:
        digest = hashlib.sha256(b"shellsim-sdk-selection-v1\0" + data + b"\0" + profile.sha256.encode()).hexdigest()
    return Port(reference, path.parent, name, version, digest, dependencies, recipe, profile)


def plan(
    root: Path, requests: Sequence[str], *, target_profile: str | None = None, default_sdk: str = "default"
) -> Graph:
    """Resolve recipe variants and SDK selection, rejecting inconsistent graphs.

    This operation reads recipe files only. Missing dependencies, cycles, version
    conflicts and target-profile mismatches fail before the driver creates a store.
    Pure recipes may omit a target profile; native profiles never change implicitly.
    """
    if isinstance(requests, (str, bytes)) or not requests or len(requests) > _MAX_NODES:
        raise ValueError("request between 1 and 512 port recipes")
    root = root.resolve(strict=True)
    roots = tuple(dict.fromkeys(_reference(request) for request in requests))
    loaded: dict[str, Port] = {}
    finished: set[str] = set()
    active: list[str] = []
    selected: dict[str, str] = {}
    ordered: list[Port] = []
    sdk_selected: str | None = None

    def visit(reference: str) -> Port:
        nonlocal sdk_selected
        if reference in active:
            raise ValueError("dependency cycle: " + " -> ".join([*active, reference]))
        if reference in finished:
            return loaded[reference]
        if len(active) >= _MAX_DEPTH or len(loaded) >= _MAX_NODES:
            raise ValueError("port dependency graph exceeds its size limit")
        active.append(reference)
        try:
            port = _read(root, reference, default_sdk)
        except (OSError, ValueError) as error:
            raise ValueError("cannot read dependency chain: " + " -> ".join(active)) from error
        if port.sdk_selection is not None:
            if sdk_selected is not None and sdk_selected != port.sdk_selection.reference:
                raise ValueError("incompatible SDK selections in dependency chain: " + " -> ".join(active))
            sdk_selected = port.sdk_selection.reference
        profile = port.recipe.get("target_profile")
        if (
            port.role != "host-tool"
            and target_profile is not None
            and profile is not None
            and profile != target_profile
        ):
            raise ValueError("target profile differs in dependency chain: " + " -> ".join(active))
        logical_name = (
            ("host" if port.role == "host-tool" else "target")
            + ":"
            + PurePosixPath(reference).parts[0]
            + "/"
            + port.name
        )
        if selected.setdefault(logical_name, reference) != reference:
            raise ValueError("conflicting recipe variants in dependency chain: " + " -> ".join(active))
        loaded[reference] = port
        for dependency in port.dependencies:
            selection = ("host" if dependency.kind == "build" else "target") + ":" + dependency.port
            previous = selected.setdefault(selection, dependency.recipe)
            if previous != dependency.recipe:
                raise ValueError("conflicting recipe variants in dependency chain: " + " -> ".join(active))
            provider = visit(dependency.recipe)
            universal = provider.recipe.get("build", {}).get("adapter") == "pure-wheel"
            host_data = provider.recipe.get("build", {}).get("adapter") == "host-wheel"
            if (dependency.kind == "build" and provider.role != "host-tool" and not universal) or (
                dependency.kind != "build"
                and provider.role == "host-tool"
                and not (host_data and port.recipe.get("build", {}).get("adapter") == "pure-wheel")
            ):
                raise ValueError("dependency role differs: " + " -> ".join([*active, dependency.recipe]))
            if (dependency.kind == "platform") != (provider.role == "target-platform"):
                raise ValueError("platform dependency role differs: " + " -> ".join([*active, dependency.recipe]))
            if provider.version != dependency.version or provider.name != PurePosixPath(dependency.port).name:
                raise ValueError("dependency name or version differs: " + " -> ".join([*active, dependency.recipe]))
        active.pop()
        finished.add(reference)
        ordered.append(port)
        return port

    for reference in roots:
        visit(reference)
    graph = Graph(roots, tuple(ordered))
    guest_graph(graph)
    return graph


def guest_graph(graph: Graph) -> Graph:
    """Select guest roots and their target/runtime closure, excluding build inputs."""
    ports = {port.reference: port for port in graph.ports}
    selected = set(graph.roots)
    pending = [(reference,) for reference in graph.roots]
    while pending:
        chain = pending.pop()
        port = ports[chain[-1]]
        if port.recipe.get("build", {}).get("adapter") == "host-wheel":
            raise ValueError("host-only provider in guest dependency chain: " + " -> ".join(chain))
        for dependency in port.dependencies:
            if dependency.kind in {"target", "runtime"} and dependency.recipe not in selected:
                selected.add(dependency.recipe)
                pending.append((*chain, dependency.recipe))
    return Graph(graph.roots, tuple(port for port in graph.ports if port.reference in selected))
