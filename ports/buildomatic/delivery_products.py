"""Emit immutable package and interpreter assets and their delivery metadata."""

from __future__ import annotations

import base64
import csv
import email
import hashlib
import io
import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.utils import canonicalize_name

from ports._support.graph import Port
from ports._support.store import relative_path
from ports.buildomatic.bytecode import BytecodeCompiler, compile_sources
from ports.buildomatic.cache import asset_path, compiler_identity, file_bundle_key, runtime_archive_key, wheel_asset_key

if TYPE_CHECKING:
    from ports.buildomatic.delivery import DeliveryStore

_MAX_ASSET = 512 * 1024**2


def _companion(source: Path, relative: str, store: DeliveryStore, *, executable: bool, scratch: Path) -> dict | None:
    """Publish ordinary Wasmtime output from declared trusted producer bytes."""
    from shellsim import _native

    if not store.exists(relative):
        with tempfile.TemporaryDirectory(prefix=".precompiled-", dir=scratch) as temporary:
            target = Path(temporary) / "module.cwasm"
            companion = _native.precompile_wasm(source.read_bytes(), executable=executable)
            if len(companion) > _MAX_ASSET:
                return None
            target.write_bytes(companion)
            store.emit(target, relative)
    size = store.size(relative)
    return {**compiler_identity(), "path": relative, "size": size} if size <= _MAX_ASSET else None


def _wheel_companions(source: Path, key: str, store: DeliveryStore, scratch: Path) -> list[dict]:
    """Publish per-module companions without changing the portable wheel or its RECORD."""
    with zipfile.ZipFile(source) as archive:
        manifests = [name for name in archive.namelist() if name.endswith(".dist-info/shellsim-native.json")]
        if not manifests:
            return []
        if len(manifests) != 1:
            raise ValueError("wheel must contain one native manifest")
        companions = []
        for entry in json.loads(archive.read(manifests[0]))["artifacts"]:
            path = relative_path(entry["path"])
            relative = asset_path(key, f"wheels/{source.name}/{path}.cwasm")
            with tempfile.TemporaryDirectory(prefix=".wheel-module-", dir=scratch) as temporary:
                module = Path(temporary) / "module.wasm"
                if not store.exists(relative):
                    module.write_bytes(archive.read(path))
                companion = _companion(module, relative, store, executable=False, scratch=scratch)
            if companion is not None:
                companions.append({**companion, "source": path})
        return companions


@dataclass(frozen=True)
class Wheel:
    """Final wheel metadata needed by a target-aware catalog consumer."""

    name: str
    version: str
    filename: str
    path: str
    size: int
    requires_python: str | None
    requires_dist: list[str]
    extras: list[str]
    tags: list[str]
    abi: str | None
    native_dependencies: list[str]


def wheel_metadata(path: Path, relative: str) -> Wheel:
    """Read metadata members only, without extracting or hashing a wheel."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            raise ValueError("wheel must contain one distribution metadata file")
        dist = members[0].removesuffix("METADATA")
        metadata = email.message_from_bytes(archive.read(members[0]))
        wheel = email.message_from_bytes(archive.read(dist + "WHEEL"))
        native = (
            json.loads(archive.read(dist + "shellsim-native.json"))
            if dist + "shellsim-native.json" in archive.namelist()
            else {}
        )
    return Wheel(
        canonicalize_name(metadata["Name"]),
        metadata["Version"],
        path.name,
        relative,
        path.stat().st_size,
        metadata["Requires-Python"],
        metadata.get_all("Requires-Dist", []),
        metadata.get_all("Provides-Extra", []),
        wheel.get_all("Tag", []),
        native.get("abi"),
        sorted({name for item in native.get("artifacts", []) for name in item["native_dependencies"]}),
    )


def _bytecode_wheel(source: Path, destination: Path, bundle: Path) -> None:
    """Keep portable members, add reproducible bytecode and reseal RECORD."""
    with zipfile.ZipFile(source) as archive:
        files = {relative_path(name): archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        modes = {member.filename: member.external_attr for member in archive.infolist()}
    info = next(name.removesuffix("METADATA") for name in files if name.endswith(".dist-info/METADATA"))
    for name in (info + "RECORD", info + "RECORD.jws", info + "RECORD.p7s"):
        files.pop(name, None)
    sources = {}
    for name, data in files.items():
        parts = Path(name).parts
        if not name.endswith(".py") or parts[0] == "bin":
            continue
        # Wheel scripts execute as source; a cache directory in scripts would
        # also be mistaken for an executable during installation.
        if len(parts) > 2 and parts[0].endswith(".data") and parts[1] == "scripts":
            continue
        sources[name] = data
    files.update(compile_sources(bundle, sources))
    record = io.StringIO(newline="")
    rows = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        rows.writerow((name, "sha256=" + digest, len(data)))
    rows.writerow((info + "RECORD", "", ""))
    files[info + "RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            member = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = modes.get(name, 0o644 << 16)
            archive.writestr(member, data)


def publish_product(
    port: Port, root: Path, build_key: str, store: DeliveryStore, *, key: str, bytecode: BytecodeCompiler | None = None
) -> dict:
    """Emit installable assets; the worker commits their completed node record."""
    record = {"reference": port.reference, "build_key": build_key, "wheels": [], "native": [], "assets": []}
    for wheel in sorted((root / "wheels").glob("*.whl")):
        relative = asset_path(wheel_asset_key(build_key, bytecode.key if bytecode else None), "wheels/" + wheel.name)
        if not store.exists(relative):
            if bytecode is None:
                store.emit(wheel, relative)
            else:
                with tempfile.TemporaryDirectory(prefix=".bytecode-wheel-", dir=root.parent) as temporary:
                    prepared = Path(temporary) / wheel.name
                    _bytecode_wheel(wheel, prepared, bytecode.bundle)
                    store.emit(prepared, relative)
        metadata = {**asdict(wheel_metadata(wheel, relative)), "size": store.size(relative)}
        companions = _wheel_companions(wheel, key, store, root.parent)
        record["wheels"].append({**metadata, **({"companions": companions} if companions else {})})
    payload = root / "payload"
    exports = port.recipe.get("exports", {})
    directories = {}
    if (root / "native/artifact.json").exists():
        from ports._support.wasm_metadata import needed_libraries

        manifest = json.loads((root / "native/artifact.json").read_text())
        exports = manifest["inputs"]["recipe"]["exports"]
        directories = manifest["inputs"]["recipe"].get("empty_directories", {})
        payload = root / "native"
        for kind, destination in (("shared_libraries", "/lib/"), ("tools", "/usr/bin/")):
            for path in exports.get(kind, []):
                source = root / "native" / relative_path(path)
                relative = asset_path(build_key, "native/" + path)
                with source.open("rb") as handle:
                    header = handle.read(4)
                    wasm = header == b"\0asm"
                if kind == "shared_libraries" and not wasm:
                    raise ValueError("shared library is not a Wasm module: " + path)
                if kind == "tools" and not wasm and not header.startswith(b"#!"):
                    raise ValueError("tool must be Wasm or a script with a shebang: " + path)
                store.emit(source, relative)
                companion = (
                    _companion(
                        source,
                        asset_path(key, f"native/{path}.cwasm"),
                        store,
                        executable=kind == "tools",
                        scratch=root.parent,
                    )
                    if wasm
                    else None
                )
                record["native"].append(
                    {
                        "name": source.name,
                        "path": relative,
                        "size": source.stat().st_size,
                        "kind": kind,
                        "destination": port.recipe.get("install", {})
                        .get("destinations", {})
                        .get(path, destination + source.name),
                        "abi": manifest["inputs"]["recipe"].get("abi"),
                        "native_dependencies": needed_libraries(source) if wasm else [],
                        **({"companions": [companion]} if companion is not None else {}),
                    }
                )
    prefix = "/" + relative_path(port.recipe.get("install", {}).get("prefix", "/usr").removeprefix("/"))
    groups = {"runtime": {}, "development": {}}
    for kind, paths in exports.items():
        if kind in {"shared_libraries", "tools"}:
            continue
        files = groups["development" if kind in {"headers", "archives", "development"} else "runtime"]
        for path in paths:
            source = payload / relative_path(path)
            destination = port.recipe.get("install", {}).get("destinations", {}).get(path)
            if destination is None:
                destination = (
                    f"/usr/share/licenses/{port.name}/" + source.name if kind == "licenses" else prefix + "/" + path
                )
            destination = relative_path(destination.removeprefix("/"))
            if destination in files and files[destination] != source:
                raise ValueError("asset installation paths conflict: " + destination)
            files[destination] = source
    for usage, files in groups.items():
        empty = directories if usage == "development" else {}
        if not files and not empty:
            continue
        relative = asset_path(file_bundle_key(build_key), f"{usage}-files.zip")
        if not store.exists(relative):
            with tempfile.TemporaryDirectory(prefix=".files-", dir=root.parent) as temporary:
                archive = Path(temporary) / "files.zip"
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
                    for destination, source in sorted(files.items()):
                        output.write(source, destination)
                    for directory in sorted(empty):
                        output.writestr(prefix.removeprefix("/") + "/" + relative_path(directory) + "/", b"")
                store.emit(archive, relative)
        size = store.size(relative)
        record["assets"].append({"path": relative, "size": size, "kind": "archive", "usage": usage})
    return record


def publish_runtime(bundle: Path, product_key: str, store: DeliveryStore, *, key: str) -> dict:
    """Emit the interpreter once, independently of package additions."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    relative = asset_path(runtime_archive_key(product_key), "runtime.zip")
    if not store.exists(relative):
        with tempfile.TemporaryDirectory(prefix=".runtime-", dir=bundle.parent) as temporary:
            archive = Path(temporary) / "runtime.zip"
            bytecode = compile_sources(
                bundle,
                {
                    path.relative_to(bundle / "rootfs").as_posix(): path.read_bytes()
                    for path in sorted((bundle / "rootfs").rglob("*.py"))
                },
            )
            manifest = {
                **manifest,
                "files": {
                    **manifest["files"],
                    **{"/" + name: hashlib.sha256(data).hexdigest() for name, data in bytecode.items()},
                },
            }
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
                metadata = {"manifest.json": json.dumps(manifest, sort_keys=True).encode()}
                metadata.update({"rootfs/" + name: data for name, data in bytecode.items()})
                for name, data in sorted(metadata.items()):
                    member = zipfile.ZipInfo("runtime/" + name, (1980, 1, 1, 0, 0, 0))
                    member.external_attr = 0o644 << 16
                    output.writestr(member, data, compress_type=zipfile.ZIP_DEFLATED)
                for path in sorted((bundle / "rootfs").rglob("*")):
                    if path.is_file() and path.relative_to(bundle / "rootfs").as_posix() not in bytecode:
                        output.write(path, "runtime/" + path.relative_to(bundle).as_posix())
            store.emit(archive, relative)
    size = store.size(relative)
    companion = _companion(
        bundle / "rootfs/usr/bin/python3.wasm",
        asset_path(key, "python3.cwasm"),
        store,
        executable=True,
        scratch=bundle.parent,
    )
    runtime = {
        "path": relative,
        "size": size,
        "abi": manifest["dynamic_abi"],
        "python_version": manifest["recipe"]["version"],
        "runtime_target": manifest["recipe"]["target"],
        **({"companions": [companion]} if companion is not None else {}),
    }
    return runtime
