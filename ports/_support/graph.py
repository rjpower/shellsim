"""Plan a bounded ports dependency graph before fetching or building anything.

Recipe references are relative to the ports tree. A directory selects recipe.json;
other variants must be named explicitly. The build driver consumes dependency-first
nodes and records their source digests alongside the eventual artifact identities.
"""

from __future__ import annotations

import hashlib
import json
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


def _read(root: Path, reference: str) -> Port:
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
    name, version = recipe.get("name"), recipe.get("version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ValueError(f"recipe must declare a name and version: {reference}")
    if PurePosixPath(name).name != name or name in {".", ".."}:
        raise ValueError(f"recipe name must be a package name: {reference}")
    Version(version)
    declarations = recipe.get("target_dependencies", [])
    if not isinstance(declarations, list) or len(declarations) > _MAX_NODES:
        raise ValueError(f"invalid dependency declarations: {reference}")
    dependencies = []
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
        dependencies.append(Dependency(port, version_pin, _reference(declaration.get("recipe", port))))
    return Port(reference, path.parent, name, version, hashlib.sha256(data).hexdigest(), tuple(dependencies), recipe)


def plan(root: Path, requests: Sequence[str], *, target_profile: str | None = None) -> Graph:
    """Resolve explicit recipe variants and reject inconsistent dependency graphs.

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

    def visit(reference: str) -> Port:
        if reference in active:
            raise ValueError("dependency cycle: " + " -> ".join([*active, reference]))
        if reference in finished:
            return loaded[reference]
        if len(active) >= _MAX_DEPTH or len(loaded) >= _MAX_NODES:
            raise ValueError("port dependency graph exceeds its size limit")
        active.append(reference)
        try:
            port = _read(root, reference)
        except (OSError, ValueError) as error:
            raise ValueError("cannot read dependency chain: " + " -> ".join(active)) from error
        profile = port.recipe.get("target_profile")
        if target_profile is not None and profile is not None and profile != target_profile:
            raise ValueError("target profile differs in dependency chain: " + " -> ".join(active))
        logical_name = PurePosixPath(reference).parts[0] + "/" + port.name
        if selected.setdefault(logical_name, reference) != reference:
            raise ValueError("conflicting recipe variants in dependency chain: " + " -> ".join(active))
        loaded[reference] = port
        for dependency in port.dependencies:
            previous = selected.setdefault(dependency.port, dependency.recipe)
            if previous != dependency.recipe:
                raise ValueError("conflicting recipe variants in dependency chain: " + " -> ".join(active))
            provider = visit(dependency.recipe)
            if provider.version != dependency.version or provider.name != PurePosixPath(dependency.port).name:
                raise ValueError("dependency name or version differs: " + " -> ".join([*active, dependency.recipe]))
        active.pop()
        finished.add(reference)
        ordered.append(port)
        return port

    for reference in roots:
        visit(reference)
    return Graph(roots, tuple(ordered))
