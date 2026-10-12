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
from ports._support.store import identity, relative_path

if TYPE_CHECKING:
    from ports.buildomatic.delivery import DeliveryStore

_MAX_ASSET = 512 * 1024**2
_MAX_WHEEL_CONTENTS = 1024**3


def _json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


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
    size = (store.local / relative).stat().st_size if store.local is not None else store.fs.size(store.key(relative))
    return {"path": relative, "size": size} if size <= _MAX_ASSET else None


def _delivery_wheel(source: Path, destination: Path) -> None:
    """Add companions for the wheel's declared extensions and update standard RECORD."""
    with zipfile.ZipFile(source) as archive:
        manifests = [name for name in archive.namelist() if name.endswith(".dist-info/shellsim-native.json")]
        if not manifests:
            destination.hardlink_to(source)
            return
        if len(manifests) != 1:
            raise ValueError("wheel must contain one native manifest")
        from shellsim import _native

        dist = manifests[0].removesuffix("shellsim-native.json")
        mapping, additions = {}, {}
        unpacked = sum(member.file_size for member in archive.infolist())
        for entry in json.loads(archive.read(manifests[0]))["artifacts"]:
            path = relative_path(entry["path"])
            companion = dist + "precompiled/" + path + ".cwasm"
            mapping[path] = companion
            data = _native.precompile_wasm(archive.read(path))
            unpacked += len(data)
            if unpacked > _MAX_WHEEL_CONTENTS:
                destination.hardlink_to(source)
                return
            additions[companion] = data
        additions[dist + "shellsim-precompiled.json"] = _json(mapping)
        unpacked += len(additions[dist + "shellsim-precompiled.json"])
        record = dist + "RECORD"
        rows = []
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            for member in archive.infolist():
                if member.filename == record:
                    continue
                if member.filename in additions:
                    raise ValueError("wheel already contains precompiled delivery members")
                data = archive.read(member)
                output.writestr(member, data)
                if not member.is_dir():
                    rows.append(_record_row(member.filename, data))
            for name, data in additions.items():
                output.writestr(name, data)
                rows.append(_record_row(name, data))
            text = io.StringIO()
            csv.writer(text, lineterminator="\n").writerows([*rows, [record, "", ""]])
            record_bytes = text.getvalue().encode()
            output.writestr(record, record_bytes)
        if destination.stat().st_size > _MAX_ASSET or unpacked + len(record_bytes) > _MAX_WHEEL_CONTENTS:
            destination.unlink()
            destination.hardlink_to(source)


def _record_row(name: str, data: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [name, "sha256=" + digest, str(len(data))]


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


def publish_product(
    port: Port, root: Path, build_key: str, store: DeliveryStore, *, record_key: str | None = None
) -> dict:
    """Emit a product's wheels/providers once, then publish their metadata record."""
    record_key = record_key or delivery_key(build_key)
    if (record := store.record(record_key)) is not None:
        return record
    record = {"reference": port.reference, "build_key": build_key, "wheels": [], "native": [], "assets": []}
    asset_key = relative_path(record_key)
    for wheel in sorted((root / "wheels").glob("*.whl")):
        relative = f"assets/delivery-v4/{asset_key}/{wheel.name}"
        with tempfile.TemporaryDirectory(prefix=".wheel-", dir=root.parent) as temporary:
            delivered = Path(temporary) / wheel.name
            _delivery_wheel(wheel, delivered)
            metadata = wheel_metadata(delivered, relative)
            store.emit(delivered, relative)
            record["wheels"].append(asdict(metadata))
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
                relative = f"assets/{build_key}/native/{path}"
                store.emit(source, relative)
                companion = _companion(
                    source,
                    f"assets/delivery-v4/{asset_key}/native/{path}.cwasm",
                    store,
                    executable=kind == "tools",
                    scratch=root.parent,
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
                        "native_dependencies": needed_libraries(source),
                        **({"precompiled": companion} if companion is not None else {}),
                    }
                )
    files = {}
    for kind, paths in exports.items():
        if kind in {"shared_libraries", "tools"}:
            continue
        for path in paths:
            source = payload / relative_path(path)
            destination = port.recipe.get("install", {}).get("destinations", {}).get(path)
            if destination is None:
                destination = (
                    f"/usr/share/licenses/{port.name}/" + source.name if kind == "licenses" else "/usr/" + path
                )
            destination = relative_path(destination.removeprefix("/"))
            if destination in files and files[destination] != source:
                raise ValueError("asset installation paths conflict: " + destination)
            files[destination] = source
    if files or directories:
        relative = f"assets/{build_key}/files.zip"
        with tempfile.TemporaryDirectory(prefix=".files-", dir=root.parent) as temporary:
            archive = Path(temporary) / "files.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
                for destination, source in sorted(files.items()):
                    output.write(source, destination)
                for directory in sorted(directories):
                    output.writestr("usr/" + relative_path(directory) + "/", b"")
            size = archive.stat().st_size
            store.emit(archive, relative)
        record["assets"].append({"path": relative, "size": size, "kind": "archive"})
    store.write("records/" + relative_path(record_key) + ".json", _json(record))
    return record


def publish_runtime(bundle: Path, key: str, store: DeliveryStore, *, record_key: str | None = None) -> dict:
    """Emit the interpreter once, independently of package additions."""
    record_key = record_key or identity({"operation": "runtime-delivery", "runtime": key, "format_version": 2})
    if (record := store.record("runtime/" + record_key)) is not None:
        return record["runtime"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    relative = f"assets/{key}/runtime.zip"
    if not store.exists(relative):
        with tempfile.TemporaryDirectory(prefix=".runtime-", dir=bundle.parent) as temporary:
            archive = Path(temporary) / "runtime.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
                for path in [bundle / "manifest.json", *sorted((bundle / "rootfs").rglob("*"))]:
                    if path.is_file():
                        output.write(path, "runtime/" + path.relative_to(bundle).as_posix())
            size = archive.stat().st_size
            store.emit(archive, relative)
    else:
        size = (
            (store.local / relative).stat().st_size if store.local is not None else store.fs.size(store.key(relative))
        )
    companion = _companion(
        bundle / "rootfs/usr/bin/python3.wasm",
        f"assets/delivery-v4/{relative_path(record_key)}/python3.cwasm",
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
        **({"precompiled": companion} if companion is not None else {}),
    }
    store.write("records/runtime/" + relative_path(record_key) + ".json", _json({"runtime": runtime}))
    return runtime


def delivery_key(product_key: str) -> str:
    return identity({"operation": "delivery", "product": product_key, "format_version": 4})
