"""Install the separately distributed C-to-Wasm compiler into a shellsim guest."""

from __future__ import annotations

import hashlib
import shlex
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Protocol

_TREE_SHA256 = {
    "tcc": "6d9aec3bf180d260f9285f6bd63f2aff550f65baf39f62a493ac126f0ec4d203",
    "wasi-sysroot": "c8db6acd30553b74e149cf9bf23387107e7587761cc930933026000af3a03e63",
}


class _RunResult(Protocol):
    def check_returncode(self) -> None: ...


class _GuestMachine(Protocol):
    def write_file(self, path: str, data: bytes, *, mode: int = 0o644) -> None: ...

    def run(self, source: str, stdin: bytes = b"") -> _RunResult: ...


def _tree_files(root: Traversable) -> tuple[tuple[str, bytes], ...]:
    pending = [("", root)]
    contents = []
    while pending:
        prefix, directory = pending.pop()
        for entry in sorted(directory.iterdir(), key=lambda child: child.name, reverse=True):
            path = f"{prefix}{entry.name}"
            if entry.is_dir():
                pending.append((f"{path}/", entry))
            else:
                contents.append((path, entry.read_bytes()))
    return tuple(sorted(contents))


def _verified_assets() -> tuple[tuple[str, tuple[tuple[str, bytes], ...]], ...]:
    assets = files("shellsim_c_toolchain").joinpath("_assets")
    bundles = []
    for name, expected_digest in _TREE_SHA256.items():
        contents = _tree_files(assets.joinpath(name))
        digest = hashlib.sha256()
        for path, payload in contents:
            digest.update(path.encode() + b"\0" + payload + b"\0")
        if digest.hexdigest() != expected_digest:
            raise ValueError(f"bundled C toolchain asset digest mismatch: {name}")
        bundles.append((name, contents))
    return tuple(bundles)


def install_c_toolchain(guest: _GuestMachine) -> None:
    """Install ``cc`` and the WASI C sysroot without granting host execution.

    The pinned files come from this distribution, never an ambient host path. The host stages
    each file directly in the bounded guest VFS.
    """

    bundles = _verified_assets()
    guest.run("test ! -e /tcc && test ! -e /wasi-sysroot").check_returncode()
    directories = {f"/{name}" for name, _ in bundles}
    for name, contents in bundles:
        for path, _ in contents:
            components = path.split("/")[:-1]
            for depth in range(1, len(components) + 1):
                directories.add(f"/{name}/{'/'.join(components[:depth])}")
    guest.run("mkdir -p " + " ".join(shlex.quote(path) for path in sorted(directories))).check_returncode()
    for name, contents in bundles:
        for path, payload in contents:
            mode = 0o755 if name == "tcc" and path == "tcc-shellsim.wasm" else 0o644
            guest.write_file(f"/{name}/{path}", payload, mode=mode)
    guest.run("rm /usr/bin/cc; ln -s /tcc/tcc-shellsim.wasm /usr/bin/cc").check_returncode()
