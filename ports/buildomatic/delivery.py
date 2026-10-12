"""Emit delivery assets once and export catalogs without reading package bytes."""

from __future__ import annotations

import errno
import html
import json
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit

from ports._support.graph import Graph, guest_graph
from ports._support.store import relative_path

logger = logging.getLogger(__name__)


class DeliveryStore:
    """Local assets use hard links; cloud access uses Rigging bucket routing."""

    def __init__(self, root: str | Path):
        from rigging.filesystem.buckets import filesystem_for
        from rigging.filesystem.storage_path import StoragePath

        path = StoragePath(str(root))
        self.fs, self.base = filesystem_for(str(path))
        self.local = Path(self.base).resolve() if path.is_local else None
        self.root = str(self.local) if self.local is not None else str(path).rstrip("/")
        self.scheme = urlsplit(str(path)).scheme
        if self.local is not None:
            self.fs = None

    def key(self, relative: str) -> str:
        return self.base.rstrip("/") + "/" + relative_path(relative)

    def exists(self, relative: str) -> bool:
        return (
            (self.local / relative_path(relative)).exists()
            if self.local is not None
            else self.fs.exists(self.key(relative))
        )

    def size(self, relative: str) -> int:
        return (
            (self.local / relative_path(relative)).stat().st_size
            if self.local is not None
            else self.fs.size(self.key(relative))
        )

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


def delivery_graph(
    graph: Graph, products: dict[str, dict], *, abi: str | None = None, runtime: dict | None = None
) -> dict:
    """Select guest records and retain the static recipe dependency edges."""
    packages, native, assets, nodes = [], [], [], []
    selected = guest_graph(graph)
    for port in selected.ports:
        if port.role in {"host-tool", "target-platform"} or port.reference not in products:
            continue
        record = products[port.reference]["delivery"]
        packages.extend({**item, "port": port.reference} for item in record["wheels"])
        native.extend({**item, "port": port.reference} for item in record["native"])
        assets.extend({**item, "port": port.reference} for item in record.get("assets", []))
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
        "assets": assets,
        **(
            {
                "runtime": {
                    **runtime,
                    "native_dependencies": [
                        item["name"]
                        for item in native
                        if item["destination"].startswith("/usr/lib/python3.13/lib-dynload/")
                    ],
                }
            }
            if runtime is not None
            else {}
        ),
    }


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

    def asset(item):
        return {
            **item,
            "url": url_prefix.rstrip("/") + "/" + quote(relative_path(item["path"]), safe="/"),
            **({"precompiled": asset(item["precompiled"])} if "precompiled" in item else {}),
        }

    for section in ("packages", "native", "assets"):
        value[section] = [asset(item) for item in graph.get(section, [])]
    if "runtime" in graph:
        value["runtime"] = asset(graph["runtime"])
    return value


def export(
    state: Path, output: str | Path, url_prefix: str, *, copy_assets: bool = False, simple: bool = False
) -> dict:
    """Write metadata and optionally promote new assets into the output store."""
    graph = json.loads((state / "graph.json").read_text())
    logger.info(
        "Exporting metadata for %d packages and %d native exports to %s",
        len(graph["packages"]),
        len(graph["native"]),
        output,
    )
    value = catalog(graph, url_prefix)
    destination = DeliveryStore(output)
    copied = 0
    if copy_assets:
        source = DeliveryStore(json.loads((state / "results.json").read_text())["prefix"])
        for item in [
            *graph["packages"],
            *graph["native"],
            *graph.get("assets", []),
            *([graph["runtime"]] if "runtime" in graph else []),
        ]:
            copied += source.promote(item["path"], destination)
            if "precompiled" in item:
                copied += source.promote(item["precompiled"]["path"], destination)
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
        pages = []
        for name, packages in projects.items():
            links = []
            for package in packages:
                requires = package["requires_python"]
                attribute = f' data-requires-python="{html.escape(requires, quote=True)}"' if requires else ""
                links.append(
                    f'<a href="{html.escape(package["url"], quote=True)}"{attribute}>{html.escape(package["filename"])}</a>'
                )
            pages.append(
                (f"simple/{relative_path(name)}/index.html", ("<!doctype html>\n" + "\n".join(links) + "\n").encode())
            )
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(destination.write, path, content) for path, content in pages]
            for future in futures:
                future.result()
    return {"packages": len(value["packages"]), "native": len(value["native"]), "assets_copied": copied}
