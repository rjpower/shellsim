"""Validate pinned pure wheels before copying or adapting their contents."""

import email
import hashlib
import io
import zipfile
from pathlib import Path, PurePosixPath


def verified_files(wheel: Path, recipe: dict, *, source: bytes | None = None) -> dict[str, bytes]:
    """Reject altered sources, unsafe members and native content before building."""
    return _verified_files(wheel, recipe, source=source, host_data=False)


def verified_host_files(wheel: Path, recipe: dict, *, source: bytes | None = None) -> dict[str, bytes]:
    """Admit unchanged host-only universal wheels, including inert launcher data."""
    if recipe.get("role") != "host-tool" or recipe["build"]["adapter"] != "host-wheel":
        raise ValueError("host wheel requires an explicit host-only recipe")
    return _verified_files(wheel, recipe, source=source, host_data=True)


def _verified_files(wheel: Path, recipe: dict, *, source: bytes | None, host_data: bool) -> dict[str, bytes]:
    if source is None:
        if wheel.stat().st_size > 64 * 1024**2:
            raise ValueError("pure wheel exceeds build bounds")
        source = wheel.read_bytes()
    if len(source) > 64 * 1024**2:
        raise ValueError("pure wheel exceeds build bounds")
    if wheel.name != recipe["source"]["filename"] or hashlib.sha256(source).hexdigest() != recipe["source"]["sha256"]:
        raise ValueError("pure wheel source identity mismatch")
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        if len(archive.infolist()) > 10000 or sum(info.file_size for info in archive.infolist()) > 64 * 1024**2:
            raise ValueError("pure wheel exceeds build bounds")
        files = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or "\0" in info.filename
                or info.filename in files
            ):
                raise ValueError("unsafe or duplicate wheel member")
            data = archive.read(info)
            if not host_data and (
                path.suffix.lower() in {".so", ".pyd", ".dll", ".dylib", ".a", ".wasm"}
                or data.startswith((b"\0asm", b"\x7fELF", b"MZ"))
            ):
                raise ValueError("pure wheel contains native code")
            files[info.filename] = data
    dist = recipe["name"].replace("-", "_") + "-" + recipe["version"] + ".dist-info/"
    if dist + "METADATA" not in files or dist + "WHEEL" not in files:
        raise ValueError("pure wheel distribution metadata is missing")
    metadata = email.message_from_bytes(files[dist + "METADATA"])
    tags = email.message_from_bytes(files[dist + "WHEEL"])
    if metadata["Name"].lower().replace("_", "-") != recipe["name"] or metadata["Version"] != recipe["version"]:
        raise ValueError("pure wheel metadata identity mismatch")
    if (
        tags["Root-Is-Purelib"] != "true"
        or any(not tag.endswith("-none-any") for tag in tags.get_all("Tag", []))
        or not tags.get_all("Tag")
    ):
        raise ValueError("wheel does not declare pure Python tags")
    if "requires_dist" in recipe and sorted(metadata.get_all("Requires-Dist", [])) != sorted(recipe["requires_dist"]):
        raise ValueError("pure wheel dependency metadata mismatch")
    return files
