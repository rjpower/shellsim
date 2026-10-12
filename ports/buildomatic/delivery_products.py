"""Emit immutable package and interpreter assets and their delivery metadata."""

from __future__ import annotations

import email
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


def _json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


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
    if (record := store.record(record_key or build_key)) is not None:
        return record
    record = {"reference": port.reference, "build_key": build_key, "wheels": [], "native": [], "assets": []}
    for wheel in sorted((root / "wheels").glob("*.whl")):
        relative = f"assets/{build_key}/{wheel.name}"
        metadata = wheel_metadata(wheel, relative)
        store.emit(wheel, relative)
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
    store.write("records/" + relative_path(record_key or build_key) + ".json", _json(record))
    return record


def publish_runtime(bundle: Path, key: str, store: DeliveryStore, *, record_key: str | None = None) -> dict:
    """Emit the interpreter once, independently of package additions."""
    if (record := store.record("runtime/" + (record_key or key))) is not None:
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
    runtime = {
        "path": relative,
        "size": size,
        "abi": manifest["dynamic_abi"],
        "python_version": manifest["recipe"]["version"],
        "runtime_target": manifest["recipe"]["target"],
    }
    store.write("records/runtime/" + relative_path(record_key or key) + ".json", _json({"runtime": runtime}))
    return runtime


def delivery_key(product_key: str) -> str:
    return identity({"operation": "delivery", "product": product_key, "format_version": 3})
