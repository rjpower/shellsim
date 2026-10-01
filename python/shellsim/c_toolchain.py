"""Install the bundled C-to-Wasm compiler and sysroot into a simulated guest."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from typing import Protocol

from ._api import RunResult

_ASSETS = {
    "tcc-shellsim-package.tar.gz": "9405d8820ea5ff6a60065a5173284631db8f871d2e6d79e8f9c3b1a7456db720",
    "sysroot-34.tar.gz": "3d637426ef54d66dfb7a03276ecbf16f925b481145573a127d244093978b65be",
}


class _GuestMachine(Protocol):
    def write_file(self, path: str, data: bytes, *, mode: int = 0o644) -> None: ...

    def run(self, source: str, stdin: bytes = b"") -> RunResult: ...


def install_c_toolchain(guest: _GuestMachine) -> None:
    """Install ``cc`` and the WASI C sysroot without granting host execution.

    All extraction and linking happen in the guest VFS under its resource limits. The pinned
    archives are read from the installed Python distribution, never from an ambient host path.
    """

    guest.run("test ! -e /work/tcc-shellsim-package.tar.gz && test ! -e /work/sysroot-34.tar.gz").check_returncode()
    for name, expected_sha256 in _ASSETS.items():
        payload = files("shellsim").joinpath("_assets", name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(f"bundled C toolchain asset digest mismatch: {name}")
        guest.write_file(f"/work/{name}", payload)
    guest.run(
        "set -e; "
        "mkdir -p /tcc /wasi-sysroot /usr/bin; "
        "tar -xzf /work/tcc-shellsim-package.tar.gz -C /tcc; "
        "tar -xzf /work/sysroot-34.tar.gz -C /wasi-sysroot; "
        "chmod +x /tcc/tcc-shellsim.wasm; "
        "rm /usr/bin/cc; "
        "ln -s /tcc/tcc-shellsim.wasm /usr/bin/cc; "
        "rm /work/tcc-shellsim-package.tar.gz /work/sysroot-34.tar.gz"
    ).check_returncode()
