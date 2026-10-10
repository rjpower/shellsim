"""Publish verified graph exports through the existing native release catalog.

Host build tools and target platforms are not guest packages. Install-only runtime
edges retain separate sealed identities from target-linked library dependencies.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from packaging.utils import canonicalize_name

from ports.native.dependencies import digest, verify_artifact

if TYPE_CHECKING:
    from ports._support.graph import Graph, Port


def _publication_size(prefix: Path, remaining: int) -> tuple[int, int]:
    """Bound directory traversal and bytes before verification or publication copies."""
    if any(path.is_symlink() for path in (prefix, *prefix.parents)):
        raise ValueError("native publication artifact has a linked parent")
    entries = files = 0
    pending = [(prefix, 0)]
    while pending:
        directory, depth = pending.pop()
        if depth > 64:
            raise ValueError("native publication artifact exceeds directory depth")
        with os.scandir(directory) as stream:
            for entry in stream:
                entries += 1
                if entries > 20_000 or entry.is_symlink():
                    raise ValueError("native publication artifact has too many or linked entries")
                if entry.is_dir(follow_symlinks=False):
                    pending.append((Path(entry.path), depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError("native publication artifact contains a special file")
                size = entry.stat(follow_symlinks=False).st_size
                if size > remaining or (entry.name == "artifact.json" and size > 1024**2):
                    raise ValueError("native publication exceeds its byte bound")
                remaining -= size
                files += 1
    return remaining, files


def installation(port: Port) -> tuple[str, str, dict[str, str]]:
    """Read a recipe's guest package alias, kind and optional export destinations."""
    declaration = port.recipe.get("install", {})
    if not isinstance(declaration, dict) or set(declaration) - {"name", "kind", "destinations", "directories"}:
        raise ValueError("native install declaration has unsupported fields")
    name = declaration.get("name", port.name)
    kind = declaration.get("kind", "build-tool" if port.role == "guest-tool" else "devel")
    destinations = declaration.get("destinations", {})
    if (
        not isinstance(name, str)
        or not name
        or canonicalize_name(name) != name
        or kind not in {"build-tool", "devel", "runtime"}
        or not isinstance(destinations, dict)
        or not isinstance(declaration.get("directories", {}), dict)
        or (port.role == "guest-tool" and kind != "build-tool")
    ):
        raise ValueError("invalid native install name, kind or destinations")
    return name, kind, destinations


def publish_native_catalog(graph: Graph, results: dict[str, Path], output: Path) -> Path | None:
    """Copy exact target artifacts and validate their public installation closure.

    The caller owns a temporary publication directory. Every exported byte is
    staged through the native installer contract, including notices. Tools use
    the existing exports.tools executable-mode contract.
    """
    from shellsim.native_packages import verify_release_catalog

    selected = [
        port
        for port in graph.ports
        if port.role not in {"host-tool", "target-platform"} and (results[port.reference] / "native").is_dir()
    ]
    if not selected:
        return None
    remaining = 2 * 1024**3
    file_count = 0
    for port in selected:
        remaining, files = _publication_size(results[port.reference] / "native", remaining)
        file_count += files
        if file_count > 10_000:
            raise ValueError("native publication exceeds its file bound")
    records = []
    aliases = {port.reference: installation(port) for port in selected}
    targets = set()
    names = set()
    owners = {}
    for port in selected:
        prefix = results[port.reference] / "native"
        manifest = verify_artifact(prefix)
        recipe = manifest["inputs"]["recipe"]
        if recipe["name"] != port.name or recipe["version"] != port.version:
            raise ValueError("graph native artifact differs from selected recipe")
        graph_recipe = manifest["inputs"].get("graph_recipe", recipe)
        if graph_recipe != port.recipe:
            raise ValueError("native artifact was built from a different graph recipe")
        name, kind, explicit = aliases[port.reference]
        if name in names:
            raise ValueError("graph native package aliases conflict")
        names.add(name)
        targets.add(recipe["target"])
        if set(explicit) - set(manifest["files"]):
            raise ValueError("native install destination names an unexported file")
        if set(port.recipe.get("install", {}).get("directories", {})) - set(manifest.get("directories", {})):
            raise ValueError("native install directory names an undeclared directory")
        destinations = {}
        for relative in manifest["files"]:
            destination = explicit.get(relative, "/usr/local/" + relative)
            path = PurePosixPath(destination) if isinstance(destination, str) else None
            if path is None or not path.is_absolute() or ".." in path.parts or path.as_posix() != destination:
                raise ValueError("native install destination must be canonical and absolute")
            mode = 0o755 if relative in recipe["exports"].get("tools", []) else 0o644
            identity = manifest["files"][relative], mode
            if destination in owners and owners[destination] != identity:
                raise ValueError("native graph install destinations conflict")
            owners[destination] = identity
            destinations[relative] = destination
        edges = []
        for edge in port.dependencies:
            if edge.kind not in {"target", "runtime"}:
                continue
            if edge.recipe not in aliases:
                raise ValueError("guest native dependency has no published artifact")
            alias, dependency_kind, _ = aliases[edge.recipe]
            edges.append(
                {"requirement": alias + "==" + edge.version, "kind": dependency_kind, "linked": edge.kind == "target"}
            )
        artifact_name = "artifacts/" + manifest["artifact_sha256"]
        destination = output / artifact_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copytree(prefix, destination)
        records.append(
            {
                "name": name,
                "version": recipe["version"],
                "kind": kind,
                "artifact": artifact_name,
                "artifact_sha256": manifest["artifact_sha256"],
                "destinations": destinations,
                "directories": {
                    name: port.recipe.get("install", {}).get("directories", {}).get(name, "/usr/local/" + name)
                    for name in manifest.get("directories", {})
                },
                "dependencies": edges,
                "recipe_name": recipe["name"],
                "recipe_port": port.reference.split("/", 1)[0] + "/" + port.name,
                "target": recipe["target"],
                "target_profile": recipe["target_profile"],
                "abi": recipe.get("abi"),
                "toolchain_sha256": digest(manifest["inputs"]["toolchain"]),
            }
        )
    if targets - {"wasm32-wasip1", "wasm32-wasip1-threads"}:
        raise ValueError("native graph publication has an unsupported target")
    catalog = output / "catalog.json"
    catalog.write_text(
        json.dumps({"format": 1, "target": "wasm32-wasip1", "packages": records}, sort_keys=True, indent=2) + "\n"
    )
    verify_release_catalog(catalog)
    return catalog
