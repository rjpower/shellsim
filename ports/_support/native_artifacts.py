"""Bounded native export sealing and dependency staging for the graph runner.

Installed paths are relative to the logical /usr/local payload. File symlinks
are normalized inside that payload; no host absolute path is followed. The
runner owns source/tool admission and publication of the resulting artifacts.
"""

import copy
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries, number, string
from ports.native.dependencies import digest, exported_paths, recipe_identity, seal_artifact, verify_artifact

MAX_ENTRIES = 16384
MAX_BYTES = 1024**3
MAX_PROVIDERS = 64
MAX_CLOSURE_BYTES = 1024**3


@dataclass(frozen=True)
class _SnapshotPath:
    data: bytes

    def read_bytes(self) -> bytes:
        return self.data


@dataclass(frozen=True)
class NativeArtifact:
    """A verified envelope and the directory that contains its exact exports."""

    prefix: Path
    manifest: Mapping


@dataclass(frozen=True)
class NativeTarget:
    """The cohort identity that every linked target dependency must match."""

    target: str
    profile: str
    abi: str | None
    toolchain: Mapping


def _relative(value: str) -> str:
    parsed = PurePosixPath(value)
    if not value or parsed.is_absolute() or ".." in parsed.parts or str(parsed) != value or value == ".":
        raise ValueError("invalid native export path")
    return value


def _no_parent_links(path: Path):
    absolute = path.absolute()
    if any(item.is_symlink() for item in (absolute, *absolute.parents)):
        raise ValueError("native staging path contains a symlink")


def _scan(root: Path, *, links: bool, byte_limit: int = MAX_BYTES):
    _no_parent_links(root)
    files, directories, symlinks = {}, set(), {}
    pending = [(root, 0)]
    count = total = 0
    if root.is_symlink() or not root.is_dir():
        raise ValueError("native payload must be a real directory")
    while pending:
        directory, depth = pending.pop()
        if depth > 64:
            raise ValueError("native payload directory depth exceeded")
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > MAX_ENTRIES:
                    raise ValueError("native payload entry bound exceeded")
                path = Path(entry.path)
                name = path.relative_to(root).as_posix()
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    if not links:
                        raise ValueError("native artifact contains a symlink")
                    symlinks[name] = os.readlink(path)
                elif stat.S_ISDIR(mode):
                    directories.add(name)
                    pending.append((path, depth + 1))
                elif stat.S_ISREG(mode):
                    size = entry.stat(follow_symlinks=False).st_size
                    total += size
                    if total > byte_limit:
                        raise ValueError("native payload byte bound exceeded")
                    data = path.read_bytes()
                    if len(data) != size:
                        raise ValueError("native payload changed while reading")
                    files[name] = data
                else:
                    raise ValueError("native payload contains a special file")
    return files, directories, symlinks


def _normalize_link(name, files, links):
    visited = set()
    while name in links:
        if name in visited or len(visited) >= 64:
            raise ValueError("native install symlink cycle")
        visited.add(name)
        target = PurePosixPath(links[name])
        if target.is_absolute():
            try:
                target = target.relative_to("/usr/local")
            except ValueError as error:
                raise ValueError("native install symlink escapes payload") from error
            parts = []
        else:
            parts = list(PurePosixPath(name).parent.parts)
        for part in target.parts:
            if part == "..":
                if not parts:
                    raise ValueError("native install symlink escapes payload")
                parts.pop()
            elif part != ".":
                parts.append(part)
        name = "/".join(parts)
    if name not in files:
        raise ValueError("native install symlink target is not a regular file")
    return files[name]


def _artifact_snapshot(artifact: NativeArtifact, byte_limit: int):
    files, _, _ = _scan(artifact.prefix, links=False, byte_limit=byte_limit)
    actual = verify_artifact(artifact.prefix)
    if actual != artifact.manifest or json.loads(files.pop("artifact.json")) != actual:
        raise ValueError("native artifact envelope changed")
    if set(files) != set(actual["files"]) or any(
        hashlib.sha256(data).hexdigest() != actual["files"][name] for name, data in files.items()
    ):
        raise ValueError("native artifact bytes changed")
    return files


def _closure(direct: Mapping[str, NativeArtifact], closure: Mapping[str, NativeArtifact], target: NativeTarget):
    providers = dict(closure)
    for name, artifact in direct.items():
        if name in providers and providers[name].manifest != artifact.manifest:
            raise ValueError("conflicting direct native provider")
        providers[name] = artifact
    snapshots, active = {}, set()

    def visit(name):
        if name in active:
            raise ValueError("native dependency cycle")
        if name in snapshots:
            return
        if len(snapshots) + len(active) >= MAX_PROVIDERS:
            raise ValueError("native dependency provider bound exceeded")
        if name not in providers:
            raise ValueError("native dependency provider is missing")
        artifact = providers[name]
        recipe = artifact.manifest["inputs"]["recipe"]
        expected_name = name.split("/")[1] if "/" in name else name
        if recipe["name"] != expected_name:
            raise ValueError("native dependency name mismatch")
        if (recipe["target"], recipe["target_profile"], recipe.get("abi")) != (
            target.target,
            target.profile,
            target.abi,
        ) or artifact.manifest["inputs"]["toolchain"] != target.toolchain:
            raise ValueError("native dependency cohort mismatch")
        requirements = recipe.get("target_dependencies", [])
        identities = artifact.manifest["inputs"]["dependency_artifacts"]
        names = [item["port"] for item in requirements]
        if len(names) != len(set(names)) or set(names) != set(identities):
            raise ValueError("native dependency edge identities differ from recipe")
        active.add(name)
        for requirement in requirements:
            child = requirement["port"]
            visit(child)
            manifest = providers[child].manifest
            if (
                manifest["inputs"]["recipe"]["version"] != requirement["version"]
                or identities[child] != manifest["artifact_sha256"]
            ):
                raise ValueError("native dependency version or artifact mismatch")
        active.remove(name)
        retained = sum(len(data) for files in snapshots.values() for data in files.values())
        snapshots[name] = _artifact_snapshot(artifact, min(MAX_BYTES, MAX_CLOSURE_BYTES - retained))
        _check_libraries(recipe, snapshots[name], providers, target.abi, require_marker=True)
        if sum(len(data) for files in snapshots.values() for data in files.values()) > MAX_CLOSURE_BYTES:
            raise ValueError("native closure byte bound exceeded")

    for name in direct:
        visit(name)
    return snapshots


def _abi(data: bytes):
    if data[:8] != b"\0asm\x01\0\0\0":
        raise ValueError("shared native export is not a Wasm module")
    markers = []
    offset = 8
    while offset < len(data):
        kind = data[offset]
        size, payload = number(data, offset + 1)
        end = payload + size
        if end > len(data):
            raise ValueError("truncated shared native Wasm section")
        if kind == 0:
            name, content = string(data, payload)
            if content > end:
                raise ValueError("invalid shared native custom section")
            if name == "shellsim.abi":
                markers.append(data[content:end].decode())
        offset = end
    if len(markers) > 1:
        raise ValueError("duplicate native ABI marker")
    return markers[0] if markers else None


def _check_libraries(recipe, files, providers, abi, *, require_marker):
    expected = sorted(
        providers[item["port"]].manifest["inputs"]["recipe"]["soname"]
        for item in recipe.get("target_dependencies", [])
        if "soname" in providers[item["port"]].manifest["inputs"]["recipe"]
    )
    libraries = recipe.get("exports", {}).get("shared_libraries", []) + recipe.get("exports", {}).get("libraries", [])
    missing = set()
    for path in libraries:
        data = files[path]
        if sorted(needed_libraries(_SnapshotPath(data))) != expected:
            raise ValueError("native library dependencies differ from declared edges")
        marker = _abi(data)
        if abi is None or (marker is not None and marker != abi) or (require_marker and marker is None):
            raise ValueError("native library ABI marker differs from cohort")
        if marker is None:
            missing.add(path)
    return missing


def merge_dependency_sysroot(
    direct: Mapping[str, NativeArtifact], closure: Mapping[str, NativeArtifact], destination: Path, target: NativeTarget
) -> Path:
    """Verify the entire linked closure before atomically staging /usr/local."""
    snapshots = _closure(direct, closure, target)
    providers = {**closure, **direct}
    empty_directories = {
        path: mode for name in snapshots for path, mode in providers[name].manifest.get("directories", {}).items()
    }
    merged = {}
    for files in snapshots.values():
        for name, data in files.items():
            _relative(name)
            if name in merged and merged[name] != data:
                raise ValueError("native dependency export collision")
            merged[name] = data
    names = set(merged)
    if any(str(parent) in names for name in names for parent in PurePosixPath(name).parents) or any(
        str(parent) in names
        for name in empty_directories
        for parent in (PurePosixPath(name), *PurePosixPath(name).parents)
    ):
        raise ValueError("native dependency file/directory collision")
    if destination.exists() or destination.is_symlink():
        raise ValueError("dependency sysroot destination already exists")
    _no_parent_links(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-dependencies-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "sysroot"
        stage.mkdir()
        for name, mode in empty_directories.items():
            path = stage / "usr/local" / name
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(mode)
        for name, data in merged.items():
            output = stage / "usr/local" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
        stage.rename(destination)
    return destination


def seal_native_install(
    recipe: Mapping,
    directory: Path,
    staging: Path,
    output: Path,
    target: NativeTarget,
    direct: Mapping[str, NativeArtifact],
    closure: Mapping[str, NativeArtifact],
    runtime: Mapping[str, NativeArtifact] | None = None,
) -> NativeArtifact:
    """Expand payload-relative exports and seal verified installed snapshots.

    `exports` lists files and `export_directories` lists directories by group.
    Both are relative to staging/usr/local. Original graph recipe identity is
    retained separately from the effective recipe containing exact file paths.
    """
    _closure(direct, closure, target)
    recipe_identity({**recipe, "build_scripts": recipe.get("build_scripts", [])}, directory)
    graph_hash = digest(recipe)
    if (recipe["target"], recipe["target_profile"], recipe.get("abi")) != (target.target, target.profile, target.abi):
        raise ValueError("native output recipe cohort mismatch")
    requirements = recipe.get("target_dependencies", [])
    if {item["port"] for item in requirements} != set(direct) or len(requirements) != len(direct):
        raise ValueError("native direct dependencies differ from recipe")
    for item in requirements:
        if direct[item["port"]].manifest["inputs"]["recipe"]["version"] != item["version"]:
            raise ValueError("native direct dependency version mismatch")
    runtime = {} if runtime is None else runtime
    requirements = recipe.get("runtime_dependencies", [])
    if {item["port"] for item in requirements} != set(runtime) or len(requirements) != len(runtime):
        raise ValueError("native runtime dependencies differ from recipe")
    _closure(runtime, closure, target)
    for item in requirements:
        if runtime[item["port"]].manifest["inputs"]["recipe"]["version"] != item["version"]:
            raise ValueError("native runtime dependency version mismatch")
    files, directories, links = _scan(staging / "usr/local", links=True)
    effective = copy.deepcopy(dict(recipe))
    groups = {group: set(values) for group, values in recipe.get("exports", {}).items()}
    for group, values in recipe.get("export_directories", {}).items():
        selected = groups.setdefault(group, set())
        for value in values:
            _relative(value)
            if value not in directories:
                raise ValueError("declared native export directory is missing")
            selected.update(name for name in files.keys() | links.keys() if name.startswith(value + "/"))
    effective.pop("export_directories", None)
    effective["exports"] = {group: sorted(values) for group, values in groups.items()}
    selected = {name: _normalize_link(name, files, links) for name in exported_paths(effective)}
    missing_markers = _check_libraries(effective, selected, direct, target.abi, require_marker=False)
    for name in effective.get("exports", {}).get("tools", []):
        data = selected[name]
        if not data.startswith(b"\0asm\x01\0\0\0"):
            continue
        marker = _abi(data)
        if target.abi is None or (marker is not None and marker != target.abi):
            raise ValueError("native tool ABI marker differs from cohort")
        if marker is None:
            missing_markers.add(name)
    empty_directories = effective.get("empty_directories", {})
    if not isinstance(empty_directories, dict) or len(empty_directories) > MAX_ENTRIES:
        raise ValueError("native empty directories must be a bounded mapping")
    for name, mode in empty_directories.items():
        _relative(name)
        if (
            mode != 0o755
            or name not in directories
            or any(item.startswith(name + "/") for item in files.keys() | links.keys() | directories - {name})
        ):
            raise ValueError("declared native empty directory is missing or nonempty")
    inputs = {
        "recipe": effective,
        "recipe_sha256": digest(effective),
        "graph_recipe": dict(recipe),
        "graph_recipe_sha256": graph_hash,
        "source_sha256": recipe["source"]["sha256"],
        "toolchain": dict(target.toolchain),
        "dependency_artifacts": {name: item.manifest["artifact_sha256"] for name, item in sorted(direct.items())},
        "runtime_artifacts": {name: item.manifest["artifact_sha256"] for name, item in sorted(runtime.items())},
    }
    if output.exists() or output.is_symlink():
        raise ValueError("native artifact output already exists")
    _no_parent_links(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-artifact-", dir=output.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        for name, mode in empty_directories.items():
            path = stage / name
            path.mkdir(parents=True)
            path.chmod(mode)
        for name, data in selected.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            if name in missing_markers:
                mark_abi(path, target.abi.encode())
        seal_artifact(stage, inputs)
        verify_artifact(stage, inputs)
        stage.rename(output)
    return NativeArtifact(output, verify_artifact(output, inputs))
