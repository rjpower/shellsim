"""Bounded, immutable `.shl` workspace packages for fresh shellsim containers."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import stat
import struct
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ._api import Container, Limits, ToolHandler

_MAX_ARCHIVE = 128 * 1024 * 1024
_MAX_PAYLOAD = 128 * 1024 * 1024
_MAX_FILE = 64 * 1024 * 1024
_MAX_MANIFEST = 1024 * 1024
_MAX_ENTRIES = 10_000
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)


@dataclass(frozen=True)
class PackageSpec:
    """Package metadata and requested ceilings; it grants no host capabilities."""

    name: str
    version: str
    entrypoint: tuple[str, ...]
    required_tools: tuple[str, ...] = ()
    limits: Limits = field(default_factory=Limits)
    working_directory: str = "/work"
    requested_clock: str = "virtual"
    requires_c_toolchain: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError("invalid package name")
        if not isinstance(self.version, str) or _NAME.fullmatch(self.version) is None:
            raise ValueError("invalid package version")
        if isinstance(self.entrypoint, (str, bytes)) or not isinstance(self.entrypoint, Sequence):
            raise TypeError("entrypoint must be a sequence of arguments")
        entrypoint = tuple(self.entrypoint)
        if not entrypoint or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in entrypoint):
            raise ValueError("entrypoint must contain nonempty text arguments")
        try:
            argument_bytes = sum(len(arg.encode("utf-8")) for arg in entrypoint)
        except UnicodeEncodeError as error:
            raise ValueError("entrypoint must be UTF-8") from error
        if len(entrypoint) > 64 or argument_bytes > 4096:
            raise ValueError("entrypoint exceeds its argument limit")
        object.__setattr__(self, "entrypoint", entrypoint)
        if isinstance(self.required_tools, (str, bytes)) or not isinstance(self.required_tools, Sequence):
            raise TypeError("required_tools must be a sequence of names")
        required_tools = tuple(self.required_tools)
        if any(not isinstance(name, str) or not name or len(name) > 256 for name in required_tools):
            raise ValueError("invalid required tool name")
        try:
            for name in required_tools:
                name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("required tool names must be UTF-8") from error
        if len(set(required_tools)) != len(required_tools):
            raise ValueError("duplicate required tool name")
        if len(required_tools) > 256:
            raise ValueError("too many required tools")
        object.__setattr__(self, "required_tools", required_tools)
        if not isinstance(self.limits, Limits):
            raise TypeError("limits must be a shellsim.Limits instance")
        if self.requested_clock not in ("virtual", "real_time"):
            raise ValueError("requested_clock must be 'virtual' or 'real_time'")
        if not isinstance(self.requires_c_toolchain, bool):
            raise TypeError("requires_c_toolchain must be bool")
        if self.working_directory == "/work":
            return
        if not isinstance(self.working_directory, str) or not self.working_directory.startswith("/work/"):
            raise ValueError("working_directory must be under /work")
        _workspace_path(self.working_directory[len("/work/") :])


@dataclass(frozen=True)
class _Entry:
    path: str
    mode: int
    data: Optional[bytes]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, request: urllib.request.Request, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> None:
        return None


class Package:
    """A validated `.shl` blob that can create independent guest containers."""

    def __init__(self, blob: bytes, spec: PackageSpec, entries: tuple[_Entry, ...]) -> None:
        self._blob = blob
        self.spec = spec
        self._entries = entries

    @classmethod
    def build_from_directory(cls, root: Union[str, os.PathLike[str]], *, spec: PackageSpec) -> Package:
        """Package a trusted host directory as `/work`, rejecting links and special files."""

        _require_spec(spec)
        root_path = Path(root)
        if not root_path.is_dir() or root_path.is_symlink():
            raise ValueError("source must be a real directory")
        entries: list[_Entry] = []
        payload = 0
        pending = [(root_path, "")]
        while pending:
            directory, prefix = pending.pop()
            with os.scandir(directory) as scan:
                children = []
                for child in scan:
                    children.append(child)
                    if len(entries) + len(children) > _MAX_ENTRIES:
                        raise ValueError("package has too many entries")
                children.sort(key=lambda child: child.name, reverse=True)
            for child in children:
                path = f"{prefix}/{child.name}" if prefix else child.name
                _workspace_path(path)
                metadata = child.stat(follow_symlinks=False)
                mode = stat.S_IMODE(metadata.st_mode)
                if stat.S_ISDIR(metadata.st_mode):
                    entries.append(_Entry(path, mode, None))
                    pending.append((Path(child.path), path))
                elif stat.S_ISREG(metadata.st_mode):
                    if metadata.st_size > _MAX_FILE:
                        raise ValueError("package file exceeds 64 MiB")
                    payload += metadata.st_size
                    if payload > _MAX_PAYLOAD:
                        raise ValueError("package payload exceeds 128 MiB")
                    with open(child.path, "rb") as source:
                        data = source.read(_MAX_FILE + 1)
                    payload += len(data) - metadata.st_size
                    if payload > _MAX_PAYLOAD:
                        raise ValueError("package payload exceeds 128 MiB")
                    entries.append(_Entry(path, mode, data))
                else:
                    raise ValueError(f"unsupported source entry: {path}")
                if len(entries) > _MAX_ENTRIES:
                    raise ValueError("package has too many entries")
        return cls._build(spec, entries)

    @classmethod
    def build_from_zip(cls, source_zip: bytes, *, spec: PackageSpec, strip_prefix: str = "") -> Package:
        """Repackage a directory ZIP; `strip_prefix` explicitly removes its enclosing folder."""

        _require_spec(spec)
        if not isinstance(source_zip, bytes) or len(source_zip) > _MAX_ARCHIVE:
            raise ValueError("source ZIP exceeds 128 MiB")
        if strip_prefix:
            _workspace_path(strip_prefix.rstrip("/"))
            strip_prefix = strip_prefix.rstrip("/") + "/"
        entries: list[_Entry] = []
        payload = 0
        with _open_zip(source_zip) as archive:
            members = archive.infolist()
            if len(members) > _MAX_ENTRIES:
                raise ValueError("source ZIP has too many entries")
            for member in members:
                _check_zip_member(member)
                name = member.filename
                if strip_prefix:
                    if name == strip_prefix:
                        continue
                    if not name.startswith(strip_prefix):
                        raise ValueError("source ZIP member is outside strip_prefix")
                    name = name[len(strip_prefix) :]
                path = name.rstrip("/") if member.is_dir() else name
                _workspace_path(path)
                if not member.is_dir():
                    if member.file_size > _MAX_FILE:
                        raise ValueError("source ZIP file exceeds 64 MiB")
                    payload += member.file_size
                    if payload > _MAX_PAYLOAD:
                        raise ValueError("source ZIP payload exceeds 128 MiB")
                unix_mode = member.external_attr >> 16
                mode = stat.S_IMODE(unix_mode) if stat.S_IFMT(unix_mode) else (0o755 if member.is_dir() else 0o644)
                data = None if member.is_dir() else _read_member(archive, member)
                entries.append(_Entry(path, mode, data))
        return cls._build(spec, entries)

    @classmethod
    def build_from_container(cls, container: Container, *, spec: PackageSpec) -> Package:
        """Capture a quiescent container's `/work` tree, excluding all running state."""

        _require_spec(spec)
        if not isinstance(container, Container):
            raise TypeError("container must be a shellsim.Container")
        snapshot = json.loads(container._native.snapshot_workspace())
        entries = []
        for item in snapshot:
            data = base64.b64decode(item["data_base64"], validate=True) if item["type"] == "file" else None
            entries.append(_Entry(item["path"], item["mode"], data))
        return cls._build(spec, entries)

    @classmethod
    def _build(cls, spec: PackageSpec, entries: Iterable[_Entry]) -> Package:
        ordered = _canonical_entries(entries)
        manifest = _manifest(spec, ordered)
        manifest_bytes = _json_bytes(manifest)
        if len(manifest_bytes) > _MAX_MANIFEST:
            raise ValueError("package manifest exceeds 1 MiB")
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            _write_member(archive, "shl.json", manifest_bytes)
            for entry in ordered:
                if entry.data is not None:
                    _write_member(archive, f"rootfs/{entry.path}", entry.data, entry.mode)
        blob = stream.getvalue()
        if len(blob) > _MAX_ARCHIVE:
            raise ValueError("package ZIP exceeds 128 MiB")
        return cls.from_bytes(blob)

    @classmethod
    def from_bytes(cls, blob: bytes, *, expected_sha256: Optional[str] = None) -> Package:
        """Validate a complete package before any guest machine is constructed."""

        if not isinstance(blob, bytes) or len(blob) > _MAX_ARCHIVE:
            raise ValueError("package ZIP exceeds 128 MiB")
        digest = hashlib.sha256(blob).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("package SHA-256 mismatch")
        with _open_zip(blob) as archive:
            members = archive.infolist()
            if len(members) > _MAX_ENTRIES + 1:
                raise ValueError("package ZIP has too many members")
            by_name: dict[str, zipfile.ZipInfo] = {}
            for member in members:
                _check_zip_member(member)
                if member.is_dir() or member.filename in by_name:
                    raise ValueError("duplicate or directory ZIP member")
                by_name[member.filename] = member
            manifest_member = by_name.get("shl.json")
            if manifest_member is None or manifest_member.file_size > _MAX_MANIFEST:
                raise ValueError("missing or oversized shl.json")
            manifest = _parse_json(_read_member(archive, manifest_member, _MAX_MANIFEST))
            spec, declared = _parse_manifest(manifest)
            expected_members = {"shl.json"}
            entries = []
            for description in declared:
                path = description["path"]
                if description["type"] == "directory":
                    entries.append(_Entry(path, description["mode"], None))
                    continue
                member_name = f"rootfs/{path}"
                member = by_name.get(member_name)
                if member is None or member.file_size != description["size"]:
                    raise ValueError(f"missing or incorrectly sized package file: {path}")
                data = _read_member(archive, member)
                if hashlib.sha256(data).hexdigest() != description["sha256"]:
                    raise ValueError(f"package file digest mismatch: {path}")
                entries.append(_Entry(path, description["mode"], data))
                expected_members.add(member_name)
            if set(by_name) != expected_members:
                raise ValueError("undeclared package ZIP member")
            ordered = _canonical_entries(entries)
            if tuple(entry.path for entry in ordered) != tuple(item["path"] for item in declared):
                raise ValueError("package inventory is not sorted")
            if spec.working_directory != "/work" and spec.working_directory[len("/work/") :] not in {
                entry.path for entry in ordered if entry.data is None
            }:
                raise ValueError("package working directory is missing")
        return cls(blob, spec, ordered)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        expected_sha256: Optional[str] = None,
        fetcher: Optional[Callable[[str], Iterable[bytes]]] = None,
    ) -> Package:
        """Fetch on the host, cap the blob, check an optional digest, and validate the package."""

        if not isinstance(url, str):
            raise ValueError("package URL must be text")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or _DIGEST.fullmatch(expected_sha256) is None
        ):
            raise ValueError("expected SHA-256 must be a lowercase hex digest")
        chunks = _fetch_https(url) if fetcher is None else fetcher(url)
        if isinstance(chunks, bytes):
            chunks = (chunks,)
        collected = io.BytesIO()
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise TypeError("fetcher must yield bytes")
            if collected.tell() + len(chunk) > _MAX_ARCHIVE:
                raise ValueError("package download exceeds 128 MiB")
            collected.write(chunk)
        return cls.from_bytes(collected.getvalue(), expected_sha256=expected_sha256)

    @classmethod
    def from_path(cls, path: Union[str, os.PathLike[str]], *, expected_sha256: Optional[str] = None) -> Package:
        """Load a local distribution without reading more than the archive limit."""

        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("package path must be a regular file")
        with source.open("rb") as stream:
            blob = stream.read(_MAX_ARCHIVE + 1)
        return cls.from_bytes(blob, expected_sha256=expected_sha256)

    @property
    def sha256(self) -> str:
        """Digest of the complete distribution blob."""

        return hashlib.sha256(self._blob).hexdigest()

    def to_bytes(self) -> bytes:
        """Return the validated `.shl` blob."""

        return self._blob

    def instantiate(
        self, *, tools: Mapping[str, ToolHandler], limits: Optional[Limits] = None, clock: str = "virtual"
    ) -> Container:
        """Create an independent machine with only the declared host tools installed."""

        if not isinstance(tools, Mapping):
            raise TypeError("tools must be a mapping")
        if limits is not None and not isinstance(limits, Limits):
            raise TypeError("limits must be a shellsim.Limits instance")
        selected = {}
        for name in self.spec.required_tools:
            handler = tools.get(name)
            if not callable(handler):
                raise ValueError(f"missing required tool: {name}")
            selected[name] = handler
        host_limits = limits or Limits()
        requested = self.spec.limits
        effective = Limits(
            **{
                name: min(getattr(requested, name), getattr(host_limits, name))
                for name in ("cpu", "memory", "disk", "output")
            }
        )
        container = Container(tools=selected, limits=effective, clock=clock)
        if self.spec.requires_c_toolchain:
            try:
                from shellsim_c_toolchain import install_c_toolchain
            except ModuleNotFoundError as error:
                if error.name != "shellsim_c_toolchain":
                    raise
                raise RuntimeError("C toolchain unavailable; install shellsim[c]") from error

            install_c_toolchain(container)
        for entry in self._entries:
            if entry.data is None:
                container.mkdir(f"/work/{entry.path}", mode=entry.mode)
            else:
                container.write_file(f"/work/{entry.path}", entry.data, mode=entry.mode)
        container._entrypoint = self.spec.entrypoint
        container._working_directory = self.spec.working_directory
        return container


def _require_spec(spec: PackageSpec) -> None:
    if not isinstance(spec, PackageSpec):
        raise TypeError("spec must be a shellsim.PackageSpec")


def _workspace_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise ValueError("invalid workspace path")
    if path.startswith("/") or "\\" in path or "\0" in path or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"invalid workspace path: {path!r}")
    try:
        path_bytes = path.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("workspace path is not UTF-8") from error
    if len(path_bytes) > 4096:
        raise ValueError("workspace path is too long")


def _canonical_entries(entries: Iterable[_Entry]) -> tuple[_Entry, ...]:
    by_path: dict[str, _Entry] = {}
    payload = 0
    for entry in entries:
        _workspace_path(entry.path)
        if entry.path in by_path:
            raise ValueError(f"duplicate package path: {entry.path}")
        if isinstance(entry.mode, bool) or not isinstance(entry.mode, int) or not 0 <= entry.mode <= 0o7777:
            raise ValueError("invalid package mode")
        if entry.data is not None:
            if not isinstance(entry.data, bytes) or len(entry.data) > _MAX_FILE:
                raise ValueError("package file exceeds 64 MiB")
            payload += len(entry.data)
            if payload > _MAX_PAYLOAD:
                raise ValueError("package payload exceeds 128 MiB")
        by_path[entry.path] = entry
        if len(by_path) > _MAX_ENTRIES:
            raise ValueError("package has too many entries")
    for path in tuple(by_path):
        parts = path.split("/")
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            current = by_path.get(parent)
            if current is None:
                by_path[parent] = _Entry(parent, 0o755, None)
            elif current.data is not None:
                raise ValueError(f"package file is a parent: {parent}")
            if len(by_path) > _MAX_ENTRIES:
                raise ValueError("package has too many entries")
    return tuple(by_path[path] for path in sorted(by_path))


def _manifest(spec: PackageSpec, entries: tuple[_Entry, ...]) -> dict[str, object]:
    described = []
    for entry in entries:
        if entry.data is None:
            described.append({"type": "directory", "path": entry.path, "mode": entry.mode})
        else:
            described.append(
                {
                    "type": "file",
                    "path": entry.path,
                    "mode": entry.mode,
                    "size": len(entry.data),
                    "sha256": hashlib.sha256(entry.data).hexdigest(),
                }
            )
    return {
        "format": 1,
        "name": spec.name,
        "version": spec.version,
        "entrypoint": list(spec.entrypoint),
        "working_directory": spec.working_directory,
        "requested_clock": spec.requested_clock,
        "required_tools": list(spec.required_tools),
        "requires_c_toolchain": spec.requires_c_toolchain,
        "limits": {name: getattr(spec.limits, name) for name in ("cpu", "memory", "disk", "output")},
        "entries": described,
    }


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    archive.writestr(info, data)


def _open_zip(blob: bytes) -> zipfile.ZipFile:
    _check_zip_directory_count(blob)
    try:
        return zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as error:
        raise ValueError("invalid package ZIP") from error


def _check_zip_directory_count(blob: bytes) -> None:
    tail = blob[-(65_535 + 22) :]
    for index in range(len(tail) - 22, -1, -1):
        if tail[index : index + 4] != b"PK\x05\x06":
            continue
        _, disk, central_disk, disk_entries, total_entries, central_size, central_offset, comment_size = (
            struct.unpack_from("<IHHHHIIH", tail, index)
        )
        if index + 22 + comment_size != len(tail):
            continue
        if disk or central_disk or disk_entries != total_entries or total_entries == 0xFFFF:
            raise ValueError("unsupported ZIP directory layout")
        if total_entries > _MAX_ENTRIES + 1:
            raise ValueError("ZIP has too many members")
        if central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
            raise ValueError("ZIP64 is unsupported")
        directory_end = len(blob) - len(tail) + index
        if central_offset + central_size != directory_end:
            raise ValueError("invalid ZIP directory offset")
        cursor = central_offset
        count = 0
        while cursor < directory_end:
            if cursor + 46 > directory_end or blob[cursor : cursor + 4] != b"PK\x01\x02":
                raise ValueError("invalid ZIP directory entry")
            name_size, extra_size, member_comment_size = struct.unpack_from("<HHH", blob, cursor + 28)
            cursor += 46 + name_size + extra_size + member_comment_size
            count += 1
            if count > _MAX_ENTRIES + 1:
                raise ValueError("ZIP has too many members")
        if cursor != directory_end or count != total_entries:
            raise ValueError("ZIP directory count mismatch")
        return
    raise ValueError("invalid package ZIP")


def _check_zip_member(member: zipfile.ZipInfo) -> None:
    if member.flag_bits & 1 or member.compress_type not in _COMPRESSION:
        raise ValueError("unsupported ZIP member")
    kind = stat.S_IFMT(member.external_attr >> 16)
    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError("ZIP links and special files are unsupported")
    if (kind == stat.S_IFDIR and not member.is_dir()) or (kind == stat.S_IFREG and member.is_dir()):
        raise ValueError("invalid ZIP directory member")
    if any(ord(character) > 127 for character in member.filename) and not member.flag_bits & 0x800:
        raise ValueError("ZIP member name must use UTF-8")


def _read_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, limit: int = _MAX_FILE) -> bytes:
    if member.file_size > limit:
        raise ValueError("ZIP member exceeds its size limit")
    try:
        with archive.open(member) as stream:
            data = stream.read(limit + 1)
            if len(data) > limit or stream.read(1):
                raise ValueError("ZIP member exceeds its size limit")
    except (zipfile.BadZipFile, EOFError) as error:
        raise ValueError("corrupt ZIP member") from error
    if len(data) != member.file_size:
        raise ValueError("ZIP member size mismatch")
    return data


def _parse_json(data: bytes) -> object:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid shl.json") from error


def _parse_manifest(manifest: object) -> tuple[PackageSpec, list[dict[str, object]]]:
    fields = {
        "format",
        "name",
        "version",
        "entrypoint",
        "working_directory",
        "requested_clock",
        "required_tools",
        "limits",
        "entries",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) not in (fields, fields | {"requires_c_toolchain"})
        or type(manifest["format"]) is not int
        or manifest["format"] != 1
    ):
        raise ValueError("unsupported package manifest")
    limit_values = manifest["limits"]
    if not isinstance(limit_values, dict) or set(limit_values) != {"cpu", "memory", "disk", "output"}:
        raise ValueError("invalid package limits")
    limits = Limits(**limit_values)
    spec = PackageSpec(
        name=manifest["name"],
        version=manifest["version"],
        entrypoint=manifest["entrypoint"],
        required_tools=manifest["required_tools"],
        limits=limits,
        working_directory=manifest["working_directory"],
        requested_clock=manifest["requested_clock"],
        requires_c_toolchain=manifest.get("requires_c_toolchain", False),
    )
    descriptions = manifest["entries"]
    if not isinstance(descriptions, list) or len(descriptions) > _MAX_ENTRIES:
        raise ValueError("invalid package inventory")
    seen = set()
    payload = 0
    for item in descriptions:
        if not isinstance(item, dict) or item.get("type") not in ("file", "directory"):
            raise ValueError("invalid package entry")
        expected = {"type", "path", "mode"}
        if item["type"] == "file":
            expected |= {"size", "sha256"}
        if set(item) != expected:
            raise ValueError("invalid package entry fields")
        _workspace_path(item["path"])
        if item["path"] in seen:
            raise ValueError("duplicate package path")
        seen.add(item["path"])
        if type(item["mode"]) is not int or not 0 <= item["mode"] <= 0o7777:
            raise ValueError("invalid package mode")
        if item["type"] == "file":
            if type(item["size"]) is not int or not 0 <= item["size"] <= _MAX_FILE:
                raise ValueError("invalid package file size")
            if not isinstance(item["sha256"], str) or _DIGEST.fullmatch(item["sha256"]) is None:
                raise ValueError("invalid package file digest")
            payload += item["size"]
            if payload > _MAX_PAYLOAD:
                raise ValueError("package payload exceeds 128 MiB")
    return spec, descriptions


def _fetch_https(url: str) -> Iterable[bytes]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("default package fetch requires an HTTPS URL")
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(url, timeout=10) as response:
        while chunk := response.read(64 * 1024):
            yield chunk
