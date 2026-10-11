"""Emit delivery assets once and export catalogs without reading package bytes."""

from __future__ import annotations

import argparse
import email
import errno
import html
import json
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import quote, unquote, urlsplit

from packaging.utils import canonicalize_name

from ports._support.graph import Graph, Port, guest_graph
from ports._support.store import relative_path
from ports.buildomatic.artifacts import write_reference


class DeliveryStore:
    """Local assets use hard links; cloud access uses Rigging bucket routing."""

    def __init__(self, root: str | Path):
        self.root = str(root).rstrip("/") or "/"
        self.scheme = urlsplit(self.root).scheme
        self.local = (
            Path(unquote(urlsplit(self.root).path))
            if self.scheme == "file"
            else Path(root)
            if not self.scheme
            else None
        )
        self.fs = None
        self.base = None
        if self.local is None:
            from rigging.filesystem.buckets import filesystem_for

            self.fs, self.base = filesystem_for(self.root)

    def key(self, relative: str) -> str:
        return self.base.rstrip("/") + "/" + relative_path(relative)

    def exists(self, relative: str) -> bool:
        return (
            (self.local / relative_path(relative)).exists()
            if self.local is not None
            else self.fs.exists(self.key(relative))
        )

    def record(self, key: str) -> dict | None:
        relative = "records/" + relative_path(key) + ".json"
        try:
            if self.local is not None:
                return json.loads((self.local / relative).read_text())
            with self.fs.open(self.key(relative), "rb") as source:
                return json.load(source)
        except FileNotFoundError:
            return None

    def write(self, relative: str, value: bytes) -> None:
        if self.local is not None:
            from ports.buildomatic.store import atomic_write

            atomic_write(self.local / relative_path(relative), value)
        else:
            self.fs.pipe_file(self.key(relative), value)

    def emit(self, source: Path, relative: str) -> None:
        if self.exists(relative):
            return
        if self.local is not None:
            destination = self.local / relative_path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.hardlink_to(source)
            except OSError as error:
                if error.errno == errno.EEXIST:
                    return
                if error.errno != errno.EXDEV:
                    raise
                with staged_file(destination) as temporary:
                    shutil.copyfile(source, temporary)
        else:
            self.fs.put_file(str(source), self.key(relative))

    def promote(self, relative: str, destination: DeliveryStore) -> bool:
        """Copy only missing assets; same-backend S3 copies stay server-side."""
        relative_path(relative)
        if destination.exists(relative):
            return False
        if self.fs is not None and self.fs is destination.fs and self.scheme == "s3":
            self.fs.cp_file(self.key(relative), destination.key(relative))
            return True
        if self.local is not None:
            destination.emit(self.local / relative, relative)
            return True
        if destination.local is not None:
            target = destination.local / relative
            with staged_file(target) as temporary:
                self.fs.get_file(self.key(relative), str(temporary))
        else:
            with self.fs.open(self.key(relative), "rb", block_size=8 * 1024**2) as source:
                with destination.fs.transaction:
                    with destination.fs.open(destination.key(relative), "wb") as target:
                        shutil.copyfileobj(source, target, length=1024**2)
        return True


@contextmanager
def staged_file(destination: Path):
    """Expose a local asset only after its transfer finishes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".asset-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


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


def publish_product(port: Port, root: Path, build_key: str, store: DeliveryStore) -> dict:
    """Emit a product's wheels/providers once, then publish their metadata record."""
    if (record := store.record(build_key)) is not None:
        return record
    record = {"reference": port.reference, "build_key": build_key, "wheels": [], "native": []}
    for wheel in sorted((root / "wheels").glob("*.whl")):
        relative = f"assets/{build_key}/{wheel.name}"
        metadata = wheel_metadata(wheel, relative)
        store.emit(wheel, relative)
        record["wheels"].append(asdict(metadata))
    if (root / "native/artifact.json").exists():
        from ports._support.wasm_metadata import needed_libraries

        manifest = json.loads((root / "native/artifact.json").read_text())
        exports = manifest["inputs"]["recipe"]["exports"]
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
    store.write("records/" + relative_path(build_key) + ".json", _json(record))
    return record


def delivery_graph(graph: Graph, products: dict[str, dict], *, abi: str | None = None) -> dict:
    """Select guest records and retain the static recipe dependency edges."""
    packages, native, nodes = [], [], []
    selected = guest_graph(graph)
    for port in selected.ports:
        if port.role in {"host-tool", "target-platform"} or port.reference not in products:
            continue
        record = products[port.reference]["delivery"]
        packages.extend(record["wheels"])
        native.extend(record["native"])
        nodes.append(
            {
                "reference": port.reference,
                "build_key": record["build_key"],
                "dependencies": [edge.recipe for edge in port.dependencies if edge.kind in {"target", "runtime"}],
            }
        )
    delivered = {node["reference"] for node in nodes}
    return {
        "schema_version": 1,
        "roots": [reference for reference in selected.roots if reference in delivered],
        "abi": abi,
        "ports": nodes,
        "packages": packages,
        "native": native,
    }


def save_local_graph(build, state: Path) -> None:
    """Record delivery metadata for existing local results without rebuilding."""
    store = DeliveryStore(state)
    products = {}
    for port in build.graph.ports:
        if port.reference not in build.results or port.role in {"host-tool", "target-platform"}:
            continue
        root = build.results[port.reference]
        products[port.reference] = {"build_key": root.name, "delivery": publish_product(port, root, root.name, store)}
    abi = next((sdk.dynamic_abi for sdk in build.sdks.values() if sdk is not None), None)
    write_reference(
        state / "results.json", {"roots": build.graph.roots, "prefix": str(state.resolve()), "products": products}
    )
    write_reference(state / "graph.json", delivery_graph(build.graph, products, abi=abi))


def _json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def catalog(graph: dict, url_prefix: str) -> dict:
    """Generate URLs from the saved graph; this function performs no storage I/O."""
    base = urlsplit(url_prefix)
    if base.scheme not in {"http", "https", "file"} or base.username or base.password or base.query or base.fragment:
        raise ValueError("URL prefix must be an HTTP(S) or file base URL")
    if base.scheme in {"http", "https"} and not base.netloc:
        raise ValueError("HTTP URL prefix must include a host")
    value = {**graph, "schema_version": 2}
    for section in ("packages", "native"):
        value[section] = [
            {**item, "url": url_prefix.rstrip("/") + "/" + quote(relative_path(item["path"]), safe="/")}
            for item in graph[section]
        ]
    return value


def export(
    state: Path, output: str | Path, url_prefix: str, *, copy_assets: bool = False, simple: bool = False
) -> dict:
    """Write metadata and optionally promote new assets into the output store."""
    graph = json.loads((state / "graph.json").read_text())
    value = catalog(graph, url_prefix)
    destination = DeliveryStore(output)
    copied = 0
    if copy_assets:
        source = DeliveryStore(json.loads((state / "results.json").read_text())["prefix"])
        for item in [*graph["packages"], *graph["native"]]:
            copied += source.promote(item["path"], destination)
    destination.write("catalog.json", _json(value))
    if simple:
        projects = {}
        for package in value["packages"]:
            projects.setdefault(package["name"], []).append(package)
        destination.write(
            "simple/index.html",
            (
                "<!doctype html>\n"
                + "\n".join(f'<a href="{quote(name)}/">{html.escape(name)}</a>' for name in sorted(projects))
                + "\n"
            ).encode(),
        )
        for name, packages in projects.items():
            links = []
            for package in packages:
                requires = package["requires_python"]
                attribute = f' data-requires-python="{html.escape(requires, quote=True)}"' if requires else ""
                links.append(
                    f'<a href="{html.escape(package["url"], quote=True)}"{attribute}>{html.escape(package["filename"])}</a>'
                )
            destination.write(
                f"simple/{relative_path(name)}/index.html", ("<!doctype html>\n" + "\n".join(links) + "\n").encode()
            )
    return {"packages": len(value["packages"]), "native": len(value["native"]), "assets_copied": copied}


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("build/ports"))
    parser.add_argument("--output", help="metadata output directory or cloud prefix; defaults to STORE/export")
    parser.add_argument("--url-prefix", required=True, help="public base URL for stored assets")
    parser.add_argument("--copy-assets", action="store_true")
    parser.add_argument("--simple", action="store_true", help="also write a Simple repository HTML index")
    args = parser.parse_args(arguments)
    result = export(
        args.store,
        args.output or args.store / "export",
        args.url_prefix,
        copy_assets=args.copy_assets,
        simple=args.simple,
    )
    print(json.dumps(result))
