"""Transport admitted SDK products without changing producer provenance.

The descriptor binds separate product inventories to streamed SHA256 blobs.
Only product roots may move: host receipts, Python environments, interpreter
closures and executable bindings retain their original absolute paths. Workers
can bind products at the original paths too, preserving path-sensitive keys.
No source checkout, retained build workspace or ambient directory is exported.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from ports._support.sdk_products import (
    MaterializedSDK,
    Receipt,
    admit_host_tools,
    admit_sdk,
    file_hash,
    read_json,
    receipt,
    tool_reference,
)

_CHUNK = 1024 * 1024
_PRODUCTS = ("sdk", "llvm", "sysroot", "cpython", "runtime")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PortableLimits:
    """Bound metadata, streaming work and expanded inventories before copying."""

    max_file_bytes: int = 512 * 1024**2
    max_bytes: int = 32 * 1024**3
    max_files: int = 300_000
    metadata_bytes: int = 64 * 1024**2

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("portable limits must be positive integers")


_DEFAULT_LIMITS = PortableLimits()


@dataclass(frozen=True)
class FileEntry:
    sha256: str
    size: int
    mode: int


@dataclass(frozen=True)
class RootInventory:
    """One selected code root; stable roots require their recorded mount path."""

    path: str
    stable: bool
    files: dict[str, FileEntry]
    symlinks: dict[str, str]
    directories: dict[str, int]


def _relative(name: str) -> Path:
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
        raise ValueError("noncanonical portable inventory path")
    return path


def _absolute(name: str) -> Path:
    path = Path(name)
    if not path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("portable root must be a canonical absolute path")
    return path


def _blob(directory: Path, digest: str) -> Path:
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError("invalid portable blob digest")
    path = directory / "blobs" / digest
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("portable blob escapes its bundle")
    return path


def _alias_target(name: str, aliases: Mapping[str, str], root: Path, stable: bool) -> str:
    """Resolve file alias chains lexically without following host filesystem links."""
    visited = set()
    while name in aliases:
        if name in visited:
            raise ValueError("portable aliases contain a cycle")
        visited.add(name)
        target = aliases[name]
        if not isinstance(target, str) or not target:
            raise ValueError("unsupported portable alias target")
        if Path(target).is_absolute():
            if not stable:
                raise ValueError("absolute product aliases cannot relocate")
            resolved = Path(os.path.normpath(target))
            if not resolved.is_relative_to(root):
                raise ValueError("portable alias escapes its mount")
            name = resolved.relative_to(root).as_posix()
        else:
            name = os.path.normpath(str(Path(name).parent / target))
            _relative(name)
    return name


def _stream(source: Path, destination: Path, expected: FileEntry, limits: PortableLimits) -> None:
    """Copy and hash bounded chunks; partial files are never accepted as blobs."""
    digest = hashlib.sha256()
    size = 0
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as incoming, destination.open("xb") as outgoing:
        info = os.fstat(incoming.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected.size:
            raise ValueError("portable blob is not the declared regular file")
        for chunk in iter(lambda: incoming.read(_CHUNK), b""):
            size += len(chunk)
            if size > expected.size or size > limits.max_file_bytes:
                raise ValueError("portable blob exceeds its declared size")
            digest.update(chunk)
            outgoing.write(chunk)
    if size != expected.size or digest.hexdigest() != expected.sha256:
        raise ValueError("portable blob changed or differs from inventory")


def _inventory(
    root: Path, files: Mapping[str, str], aliases: Mapping[str, str], *, stable: bool = False
) -> RootInventory:
    """Select declared files and internal alias targets, excluding unrelated work."""
    root = root.resolve()
    regular = {}
    links = dict(aliases)
    directories = {".": stat.S_IMODE(root.stat().st_mode) & 0o777}
    pending = list(files) + list(aliases)
    seen = set()
    while pending:
        name = pending.pop()
        _relative(name)
        if name in seen:
            continue
        seen.add(name)
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("portable alias escapes its product root")
        if path.is_symlink():
            links[name] = os.readlink(path)
            if not path.resolve().is_file():
                raise ValueError("portable aliases must resolve to regular files")
            target = Path(links[name])
            pending.append(
                target.relative_to(root).as_posix()
                if target.is_absolute()
                else os.path.normpath(str(Path(name).parent / target))
            )
            if name in files and file_hash(path) != files[name]:
                raise ValueError("portable alias bytes differ")
        else:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("portable inventory requires regular files")
            digest = file_hash(path)
            if name in files and digest != files[name]:
                raise ValueError("portable inventory differs from its receipt")
            regular[name] = FileEntry(digest, info.st_size, stat.S_IMODE(info.st_mode) & 0o777)
    for name, target in links.items():
        _relative(name)
        path = root / name
        if not path.is_symlink() or os.readlink(path) != target:
            raise ValueError("portable alias differs from its receipt")
        # Host mounts retain their original absolute alias targets.
        if (Path(target).is_absolute() and not stable) or not path.resolve().is_relative_to(root) or not path.exists():
            raise ValueError("unsupported portable alias")
        canonical = path.resolve().relative_to(root).as_posix()
        if canonical not in regular:
            raise ValueError("portable alias target is outside selected inventory")
    for name in (*regular, *links):
        for parent in _relative(name).parents:
            if parent == Path("."):
                break
            path = root / parent
            if path.is_symlink():
                raise ValueError("portable inventory has a symlink parent")
            directories[parent.as_posix()] = stat.S_IMODE(path.stat().st_mode) & 0o777
    return RootInventory(str(root), stable, regular, links, directories)


def _target_bindings(context: MaterializedSDK) -> dict:
    bindings = {}
    for name, tool in context.target_tools.items():
        for provider, product in (("llvm", context.llvm), ("sdk", context.sdk)):
            try:
                relative = tool.path.relative_to(product.root).as_posix()
            except ValueError:
                continue
            bindings[name] = {"provider": provider, "path": relative, "sha256": tool.sha256}
            break
        else:
            raise ValueError("target executable has no admitted product")
    return bindings


def _admit(
    products: Mapping[str, Receipt | None], hosts: dict, targets: dict, target: str, abi: str
) -> MaterializedSDK:
    return admit_sdk(
        products["sdk"],
        products["llvm"],
        products["sysroot"],
        products["cpython"],
        products["runtime"],
        admit_host_tools(Path("/"), hosts),
        targets,
        target=target,
        abi=abi,
    )


def export_sdk(context: MaterializedSDK, destination: Path, *, limits: PortableLimits = _DEFAULT_LIMITS) -> Path:
    """Re-admit inputs and export exact receipt bytes and selected code closures.

    ``destination`` must be absent. Its descriptor and blobs may be transported
    by any directory bundler. Receipts are never serialized anew. Host roots
    are recorded at their stable absolute paths, including receipt locations.
    """
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    products = dict(
        zip(_PRODUCTS, (context.sdk, context.llvm, context.sysroot, context.cpython_manifest, context.runtime))
    )
    hosts = {name: tool_reference(tool) for name, tool in context.host_tools.items()}
    targets = _target_bindings(context)
    for product in products.values():
        if product is not None and (
            file_hash(product.path) != product.sha256 or read_json(product.path) != product.contents
        ):
            raise ValueError("SDK receipt bytes differ from admitted provenance")
    admitted = _admit(products, hosts, targets, context.target, context.dynamic_abi)
    if admitted.identity != context.identity:
        raise ValueError("SDK identity differs from admitted provenance")
    roots = {}
    references = {}
    for name, product in products.items():
        if product is None:
            references[name] = None
            continue
        if name == "sdk":
            files, aliases = product.contents, {}
        elif name in {"llvm", "sysroot"}:
            files, aliases = product.contents["artifacts"], product.contents.get("symlinks", {})
        else:
            files = {"rootfs/" + item.lstrip("/"): digest for item, digest in product.contents["files"].items()}
            if name == "cpython":
                files.update(product.contents["build_profile"]["headers"])
            aliases = {}
        selected = _inventory(product.root, files, aliases)
        receipt_name = (
            product.path.relative_to(product.root).as_posix()
            if product.path.is_relative_to(product.root)
            else "manifest.json"
        )
        _relative(receipt_name)
        if receipt_name in selected.files or receipt_name in selected.symlinks:
            raise ValueError("product receipt overlaps a declared artifact")
        entry = FileEntry(product.sha256, product.path.stat().st_size, 0o444)
        selected.files[receipt_name] = entry
        for parent in Path(receipt_name).parents:
            selected.directories.setdefault(parent.as_posix(), 0o555)
        roots[name] = selected
        references[name] = {
            "root": name,
            "manifest": receipt_name,
            "sha256": product.sha256,
            "original_manifest": str(product.path),
        }
    stable_paths = {}

    def stable(root: Path, files: Mapping[str, str], aliases: Mapping[str, str]) -> None:
        selected = _inventory(root, files, aliases, stable=True)
        key = stable_paths.setdefault(selected.path, "host-" + str(len(stable_paths)))
        existing = roots.get(key)
        if existing is not None:
            if any(existing.files[item] != entry for item, entry in selected.files.items() if item in existing.files):
                raise ValueError("conflicting stable host inventories")
            selected.files.update(existing.files)
            selected.symlinks.update(existing.symlinks)
            selected.directories.update(existing.directories)
        roots[key] = RootInventory(selected.path, True, selected.files, selected.symlinks, selected.directories)

    for name, tool in sorted(context.host_tools.items()):
        if tool.receipt_path is None:
            stable(tool.path.parent, {tool.path.name: tool.sha256}, {})
            continue
        proof = read_json(tool.receipt_path)
        stable(tool.receipt_path.parent, {tool.receipt_path.name: tool.receipt_sha256}, {})
        if name == "uv":
            stable(tool.path.parent, {tool.path.name: tool.sha256}, {})
            continue
        stable((tool.receipt_path.parent / proof["root"]).resolve(), proof["files"], proof.get("symlinks", {}))
        source = proof["source"]
        if "interpreter" in source:
            interpreter = source["interpreter"]
            stable(Path(interpreter["root"]), interpreter["files"], interpreter["symlinks"])
            source = interpreter
        if "base_python" in source:
            base = source["base_python"]
            stable(Path(base["root"]), base["files"], base["symlinks"])
    descriptor = {
        "schema_version": 1,
        "identity": context.identity,
        "target": context.target,
        "dynamic_abi": context.dynamic_abi,
        "products": references,
        "host_tools": hosts,
        "target_tools": targets,
        "roots": {name: asdict(root) for name, root in roots.items()},
    }
    _validate(descriptor, limits)
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if len(payload) > limits.metadata_bytes:
        raise ValueError("portable descriptor exceeds metadata bound")
    for root in roots.values():
        if destination.is_relative_to(Path(root.path)):
            raise ValueError("export destination overlaps an input root")
    (destination / "blobs").mkdir(parents=True)
    for name, root in roots.items():
        for relative, entry in root.files.items():
            blob = _blob(destination, entry.sha256)
            if blob.exists():
                continue
            reference = references.get(name)
            source = (
                Path(reference["original_manifest"])
                if reference and relative == reference["manifest"]
                else Path(root.path) / relative
            )
            _stream(source, blob, entry, limits)
            blob.chmod(0o444)
    path = destination / "sdk.json"
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def _validate(value: dict, limits: PortableLimits) -> dict[str, RootInventory]:
    if (
        set(value)
        != {"schema_version", "identity", "target", "dynamic_abi", "products", "host_tools", "target_tools", "roots"}
        or value["schema_version"] != 1
    ):
        raise ValueError("unsupported portable SDK descriptor")
    if set(value["products"]) != set(_PRODUCTS):
        raise ValueError("portable product roles differ")
    roots = {}
    count = size = 0
    for name, item in value["roots"].items():
        _relative(name)
        if len(Path(name).parts) != 1 or set(item) != {"path", "stable", "files", "symlinks", "directories"}:
            raise ValueError("portable root fields differ")
        _absolute(item["path"])
        if type(item["stable"]) is not bool:
            raise ValueError("portable root stability must be explicit")
        files = {}
        for relative, entry in item["files"].items():
            _relative(relative)
            if set(entry) != {"sha256", "size", "mode"} or not _DIGEST.fullmatch(entry["sha256"]):
                raise ValueError("portable file fields differ")
            if type(entry["size"]) is not int or not 0 <= entry["size"] <= limits.max_file_bytes:
                raise ValueError("portable file exceeds size bound")
            files[relative] = FileEntry(**entry)
            size += entry["size"]
        directories = item["directories"]
        if "." not in directories:
            raise ValueError("portable root mode is missing")
        aliases = item["symlinks"]
        if set(files) & set(aliases) or (set(files) | set(aliases)) & set(directories):
            raise ValueError("portable inventory entries overlap")
        for relative, mode in (*[(key, entry.mode) for key, entry in files.items()], *directories.items()):
            if relative != ".":
                _relative(relative)
            if type(mode) is not int or not 0 <= mode <= 0o777:
                raise ValueError("unsupported portable permission mode")
        for relative in aliases:
            _relative(relative)
            normalized = _alias_target(relative, aliases, Path(item["path"]), item["stable"])
            if normalized not in files:
                raise ValueError("portable alias target is not a selected regular file")
        for relative in (*files, *aliases, *directories):
            if relative == ".":
                continue
            if any(parent.as_posix() not in directories for parent in Path(relative).parents):
                raise ValueError("portable entry has an undeclared parent directory")
        count += len(files) + len(aliases) + len(directories)
        if count > limits.max_files or size > limits.max_bytes:
            raise ValueError("portable inventory exceeds resource bounds")
        roots[name] = RootInventory(item["path"], item["stable"], files, aliases, directories)
        if not item["stable"] and name not in _PRODUCTS:
            raise ValueError("unknown portable product root")
    for name, reference in value["products"].items():
        if reference is None:
            if name in {"sdk", "llvm", "sysroot"}:
                raise ValueError("required portable product is absent")
            continue
        if (
            set(reference) != {"root", "manifest", "sha256", "original_manifest"}
            or reference["root"] != name
            or name not in roots
            or roots[name].stable
        ):
            raise ValueError("portable receipt binding differs")
        _absolute(reference["original_manifest"])
        if (
            reference["manifest"] not in roots[name].files
            or roots[name].files[reference["manifest"]].sha256 != reference["sha256"]
        ):
            raise ValueError("portable receipt is outside its inventory")
    for item in value["host_tools"].values():
        _absolute(item["path"])
        _host_entry(roots, Path(item["path"]), item["sha256"])
        if item["receipt"] is not None:
            _absolute(item["receipt"]["path"])
            _host_entry(roots, Path(item["receipt"]["path"]), item["receipt"]["sha256"])
    return roots


def _host_entry(roots: Mapping[str, RootInventory], path: Path, digest: str) -> None:
    for root in roots.values():
        if not root.stable or not path.is_relative_to(Path(root.path)):
            continue
        name = path.relative_to(Path(root.path)).as_posix()
        name = _alias_target(name, root.symlinks, Path(root.path), True)
        if name in root.files and root.files[name].sha256 == digest:
            return
    raise ValueError("host binding is outside the exported stable inventory")


def _verify_host_closures(value: dict, roots: Mapping[str, RootInventory], directory: Path) -> None:
    """Require exported package and interpreter closures, even on local imports."""
    for name, tool in value["host_tools"].items():
        if tool["receipt"] is None or name == "uv":
            continue
        producer = read_json(_blob(directory, tool["receipt"]["sha256"]))
        closures = [((Path(tool["receipt"]["path"]).parent / producer["root"]).resolve(), producer)]
        source = producer["source"]
        if "interpreter" in source:
            source = source["interpreter"]
            closures.append((Path(source["root"]), source))
        if "base_python" in source:
            base = source["base_python"]
            closures.append((Path(base["root"]), base))
        for path, closure in closures:
            matching = [root for root in roots.values() if root.stable and Path(root.path) == path]
            if len(matching) != 1:
                raise ValueError("host package closure is missing from export")
            root = matching[0]
            for relative, digest in closure["files"].items():
                _host_entry(roots, path / relative, digest)
            if any(root.symlinks.get(relative) != target for relative, target in closure.get("symlinks", {}).items()):
                raise ValueError("host package alias closure differs")


def _verify(root: Path, inventory: RootInventory, *, exact: bool) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("portable mount root is not a real directory")
    for name, entry in inventory.files.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or file_hash(path) != entry.sha256
            or path.stat().st_size != entry.size
        ):
            raise ValueError("portable mount file differs")
        if stat.S_IMODE(path.stat().st_mode) & 0o555 != entry.mode & 0o555:
            raise ValueError("portable mount executable mode differs")
    for name, target in inventory.symlinks.items():
        path = root / name
        if not path.is_symlink() or os.readlink(path) != target or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("portable mount alias differs")
    for name in inventory.directories:
        path = root / name
        if not path.is_dir() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("portable mount directory differs")
    if exact:
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*")}
        expected = set(inventory.files) | set(inventory.symlinks) | (set(inventory.directories) - {"."})
        if actual != expected:
            raise ValueError("portable product mount contains unrelated files")


def original_root_bindings(descriptor: Path, *, limits: PortableLimits = _DEFAULT_LIMITS) -> Mapping[str, Path]:
    """Return validated original mount paths for every product and host root.

    Passing this mapping to ``import_sdk(..., bindings=...)`` explicitly allows
    restoring missing host mounts at their recorded paths. It also preserves
    product paths used in build flags. Existing mounts are verified unchanged.
    The descriptor must come from the enclosing trusted input bundle.
    """
    if descriptor.stat().st_size > limits.metadata_bytes:
        raise ValueError("portable descriptor exceeds metadata bound")
    roots = _validate(json.loads(descriptor.read_text()), limits)
    return {name: Path(root.path) for name, root in roots.items()}


def import_sdk(
    descriptor: Path,
    destination: Path,
    *,
    bindings: Mapping[str, Path] | None = None,
    original_bindings: bool = False,
    limits: PortableLimits = _DEFAULT_LIMITS,
) -> MaterializedSDK:
    """Restore products then admit them, preserving all immutable receipt bytes.

    Products default to ``destination/<role>``; ``original_bindings=True`` uses
    the recorded product roots, preserving path-sensitive build cache identity.
    ``bindings`` overrides individual roots by descriptor name. Stable host
    roots cannot relocate, and are restored only when explicitly bound at their
    original path. ``original_root_bindings`` supplies those explicit bindings
    for a fresh worker. Existing mounts must verify and are never overwritten.
    Restored trees are read-only. Failed imports may leave a rejected tree for
    diagnosis; they never return a partially admitted context.
    """
    descriptor = descriptor.resolve()
    if descriptor.stat().st_size > limits.metadata_bytes:
        raise ValueError("portable descriptor exceeds metadata bound")
    value = json.loads(descriptor.read_text())
    inventories = _validate(value, limits)
    bindings = dict(bindings or {})
    if set(bindings) - set(inventories):
        raise ValueError("unknown portable root binding")
    locations = {}
    receipt_paths = {}
    for name, inventory in inventories.items():
        original = Path(inventory.path)
        path = Path(
            bindings.get(name, original if inventory.stable or original_bindings else destination.absolute() / name)
        ).absolute()
        _absolute(str(path))
        if path.resolve() != path:
            raise ValueError("portable mount binding traverses a symlink")
        if inventory.stable and path != original:
            raise ValueError("host tool relocation is unsupported")
        locations[name] = path
        if path.exists() or path.is_symlink():
            selected = inventory
            reference = value["products"].get(name)
            if reference and path == original and not Path(reference["original_manifest"]).is_relative_to(original):
                # A tooling receipt may originally live outside its code root.
                # Reuse the transported receipt without adding files to that mount.
                receipt_paths[name] = _blob(descriptor.parent, reference["sha256"])
                selected = RootInventory(
                    inventory.path,
                    inventory.stable,
                    {key: entry for key, entry in inventory.files.items() if key != reference["manifest"]},
                    inventory.symlinks,
                    inventory.directories,
                )
            _verify(path, selected, exact=not inventory.stable and path != original)
        elif inventory.stable and name not in bindings:
            raise ValueError("stable host mount is missing; bind its original path explicitly")
    new = {name: path for name, path in locations.items() if not path.exists()}
    for name, path in new.items():
        if any(
            path == other or path.is_relative_to(other) or other.is_relative_to(path)
            for key, other in locations.items()
            if key != name
        ):
            raise ValueError("restored portable roots overlap")
        if descriptor.is_relative_to(path):
            raise ValueError("portable mount overlaps the input bundle")
    # Check every blob before mutating any mount, including reused stable roots.
    for inventory in inventories.values():
        for entry in inventory.files.values():
            blob = _blob(descriptor.parent, entry.sha256)
            if blob.stat().st_size != entry.size or file_hash(blob) != entry.sha256:
                raise ValueError("portable input blob differs")
    _verify_host_closures(value, inventories, descriptor.parent)
    for name, path in new.items():
        inventory = inventories[name]
        path.mkdir(parents=True)
        for relative in sorted(set(inventory.directories) - {"."}, key=lambda item: (len(Path(item).parts), item)):
            (path / relative).mkdir()
        for relative, entry in inventory.files.items():
            output = path / relative
            _stream(_blob(descriptor.parent, entry.sha256), output, entry, limits)
            output.chmod(entry.mode & ~0o222)
        for relative, target in inventory.symlinks.items():
            (path / relative).symlink_to(target)
        for relative, mode in sorted(
            inventory.directories.items(), key=lambda item: len(Path(item[0]).parts), reverse=True
        ):
            (path / relative).chmod(mode & ~0o222)
        _verify(path, inventory, exact=not inventory.stable)
    products = {}
    for name, reference in value["products"].items():
        products[name] = (
            None
            if reference is None
            else receipt(
                Path("/"),
                {
                    "root": str(locations[reference["root"]]),
                    "manifest": str(receipt_paths.get(name, locations[reference["root"]] / reference["manifest"])),
                    "sha256": reference["sha256"],
                },
            )
        )
    result = _admit(products, value["host_tools"], value["target_tools"], value["target"], value["dynamic_abi"])
    if result.identity != value["identity"]:
        raise ValueError("imported SDK identity differs")
    return result


def descriptor_digest(descriptor: Path) -> str:
    """Identify exact descriptor bytes for the enclosing trusted build request."""
    return file_hash(descriptor)
