"""Transport selected recipe and adapter source files for ordinary worker actions."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Mapping

from ports._support.graph import Graph
from ports._support.node_build import _admit_recipe
from ports._support.store import file_hash, relative_path
from ports.api import implementation

PRODUCT_SYSTEMS = frozenset({"llvm-host", "wasi-sysroot", "sdk-tooling", "cpython-threaded", "uv-host"})


def code_files(
    ports: Path, graph: Graph, *, portable: bool = False, extra_files: tuple[str, ...] = ()
) -> dict[str, Path]:
    """Enumerate code, recipes and declared inputs, without scanning a worktree.

    Python imports expand the shared runtime closure. Dynamic port helpers come
    only from implementation declarations. Optional backend imports in the CLI
    do not belong to worker code. Every selected input is regular and unlinked.
    """
    shared = Path(__file__).resolve().parents[1]
    ports = ports.resolve()
    files: dict[str, Path] = {}

    def add(name: str, path: Path, boundary: Path) -> None:
        name = relative_path(name)
        if not path.resolve(strict=True).is_relative_to(boundary):
            raise ValueError("code input escapes its admitted tree")
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(boundary)):
            raise ValueError("linked code inputs are unsupported")
        file_hash(path, limit=16 * 1024**2)
        if name in files and file_hash(files[name]) != file_hash(path):
            raise ValueError("selected code conflicts with the shared runtime: " + name)
        files[name] = path

    for name in ("api.py", "_support/node_build.py", "_support/store.py", "_support/local_sources.py", *extra_files):
        add(name, shared / name, shared)
    if portable:
        add("buildomatic/portable.py", shared / "buildomatic/portable.py", shared)
    for port in graph.ports:
        _admit_recipe(port)
        recipe_path = port.reference.partition(":")[0]
        add(recipe_path, ports / recipe_path, ports)
        if port.sdk_selection is not None:
            add(port.sdk_selection.reference, ports / port.sdk_selection.reference, ports)
        for name in implementation(port):
            if name == "builder":
                path = port.directory / "build.py"
                add(path.relative_to(ports).as_posix(), path, ports)
            elif name.startswith("local:"):
                name = name.removeprefix("local:")
                add(name, ports / name, ports)
            else:
                add(name, shared / name, shared)
        for patch in port.recipe.get("patches", []):
            path = port.directory / relative_path(patch["file"])
            add(path.relative_to(ports).as_posix(), path, ports)
        for item in port.recipe.get("build_scripts", []):
            path = port.directory / relative_path(item["file"])
            add(path.relative_to(ports).as_posix(), path, ports)
        for name in port.recipe.get("inputs", {}):
            add(name, shared / relative_path(name), shared)
        for item in port.recipe.get("source", {}).get("files", []):
            add(item["path"], ports / relative_path(item["path"]), ports)
    # The default alias is needed when a recipe selected the named default SDK.
    if any(port.sdk_selection for port in graph.ports) and (ports / "sdks/default.json").exists():
        add("sdks/default.json", ports / "sdks/default.json", ports)
    pending, inspected = list(files), set()
    while pending:
        name = pending.pop()
        if name in inspected or not name.endswith(".py"):
            continue
        inspected.add(name)
        package = "ports." + name.removesuffix(".py").replace("/", ".")
        for node in ast.walk(ast.parse(files[name].read_bytes())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parent = package.split(".")[: -node.level]
                    module = ".".join([*parent, module]).rstrip(".")
                modules = [module, *(module + "." + alias.name for alias in node.names)]
            for module in modules:
                if not module.startswith("ports."):
                    continue
                if module.startswith("ports.buildomatic") and not (portable and module == "ports.buildomatic.portable"):
                    continue
                if ".tests" in module:
                    raise ValueError("worker code cannot import test modules")
                relative = module.removeprefix("ports.").replace(".", "/")
                candidates = (relative + ".py", relative + "/__init__.py")
                for candidate in candidates:
                    if candidate in files:
                        pending.append(candidate)
                        break
                    path = shared / candidate
                    if path.is_file():
                        add(candidate, path, shared)
                        pending.append(candidate)
                        break
    if "_support/producer_migration.py" in files:
        add("_support/producer-migration-v1.json", shared / "_support/producer-migration-v1.json", shared)
    if "python/cpython/threaded.py" in files:
        add("toolchain/wasi_process/recipe.json", ports / "toolchain/wasi_process/recipe.json", ports)
    return dict(sorted(files.items()))


def _stage_code(files: Mapping[str, Path], destination: Path) -> None:
    """Stage the selected source files and preserve executable modes."""
    for name, source in files.items():
        output = destination / "ports" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        output.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
