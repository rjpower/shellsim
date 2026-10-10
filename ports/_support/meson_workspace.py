"""Retain verified Meson compilation state independently of packaging results.

Only explicit workspaces participate. Existing unrecorded trees are rejected;
source, tools, products and configure inputs must still match on every resume.
Immutable graph results remain independently keyed and sealed by the runner.
"""

from __future__ import annotations

import fcntl
import json
import shutil
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator, Mapping

from ports._support import native_adapters
from ports._support.native_adapters import NativeBuildContext
from ports._support.store import file_hash, identity


def _inventory(root: Path) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("retained Meson input is not an admitted directory")
    files, directories = {}, []
    size = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("retained Meson input contains a symlink")
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(name)
        elif path.is_file():
            size += path.stat().st_size
            if size > 2 * 1024**3 or len(files) >= 100_000:
                raise ValueError("retained Meson input exceeds inventory bounds")
            files[name] = {"sha256": file_hash(path), "executable": bool(path.stat().st_mode & 0o111)}
        else:
            raise ValueError("retained Meson input is not a regular file or directory")
    return {"files": files, "directories": directories}


@contextmanager
def retained_meson(
    context: NativeBuildContext,
    ninja_directory: Path,
    configuration: Mapping,
    products: Mapping,
    host_tools: Mapping,
    *,
    driver_inputs: Callable[[NativeBuildContext], Mapping] | None = None,
) -> Iterator[NativeBuildContext]:
    """Lock and admit an explicit Ninja tree, preserving stable input paths.

    Packaging selections and implementation identity stay in the result cache.
    Configuration, actual target products and host code remain compilation
    inputs. A mismatch rejects the requested workspace without deleting it.
    """
    ninja_directory = ninja_directory.resolve()
    if ninja_directory.name != "meson-build":
        raise ValueError("retained Meson Ninja directory must be named meson-build")
    build = ninja_directory.parent
    root = build.parent
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".meson-workspace.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source, dependencies = root / "source", root / "dependencies"
        old_prefix = str(context.dependency_sysroot)
        stable = replace(
            context,
            retained_workspace=True,
            source=source,
            build=build,
            dependency_sysroot=dependencies,
            dependencies=dict.fromkeys(context.dependencies, dependencies / "usr/local"),
            compiler_flags=tuple(flag.replace(old_prefix, str(dependencies)) for flag in context.compiler_flags),
            linker_flags=tuple(flag.replace(old_prefix, str(dependencies)) for flag in context.linker_flags),
        )
        inputs = {
            "driver": (
                driver_inputs(stable)
                if driver_inputs is not None
                else native_adapters.compilation_driver_inputs(stable, configuration.get("configure_environment", {}))
            ),
            "source": _inventory(context.source),
            "dependencies": _inventory(context.dependency_sysroot),
            "configuration": dict(configuration),
            "products": dict(products),
            "host_tools": dict(host_tools),
            "target_tools": {name: str(path) for name, path in context.target_tools.items()},
            "flags": {
                "compiler": stable.compiler_flags,
                "linker": stable.linker_flags,
                "shared": stable.shared_library_flags,
                "shared_inputs": [str(path) for path in stable.shared_library_inputs],
                "executable": stable.executable_flags,
            },
        }
        # Normalize tuples exactly as the persisted JSON representation.
        inputs = json.loads(json.dumps(inputs, sort_keys=True))
        receipt = root / ".meson-workspace.json"
        if receipt.is_symlink():
            raise ValueError("retained Meson receipt cannot be a symlink")
        if receipt.exists():
            if receipt.stat().st_size > 32 * 1024**2:
                raise ValueError("retained Meson receipt exceeds its bound")
            recorded = json.loads(receipt.read_text())
            if recorded != {"schema_version": 2, "inputs": inputs, "identity": identity(inputs)}:
                print("ports: Meson workspace rejected: compilation inputs differ", file=sys.stderr, flush=True)
                raise ValueError("retained Meson compilation inputs differ")
            if _inventory(source) != inputs["source"] or _inventory(dependencies) != inputs["dependencies"]:
                raise ValueError("retained Meson input bytes differ")
            print(f"ports: Meson workspace reused: {ninja_directory}", file=sys.stderr, flush=True)
        else:
            if build.exists() or source.exists() or dependencies.exists():
                raise ValueError("existing Meson tree has no verified workspace receipt")
            shutil.copytree(context.source, source)
            shutil.copytree(context.dependency_sysroot, dependencies)
            receipt.write_text(
                json.dumps({"schema_version": 2, "inputs": inputs, "identity": identity(inputs)}, sort_keys=True) + "\n"
            )
            print(f"ports: Meson workspace initialized: {ninja_directory}", file=sys.stderr, flush=True)
        yield stable
