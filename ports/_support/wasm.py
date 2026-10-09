"""Small build helpers shared by WASI ports and runtime test fixtures."""

import subprocess
from pathlib import Path


def leb(value: int) -> bytes:
    """Encode an unsigned section length using Wasm's LEB128 representation."""
    if value < 0:
        raise ValueError("Wasm section lengths must be nonnegative")
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def mark_abi(path: Path, abi: bytes) -> None:
    """Attach the explicit trusted ABI identity checked by the runtime."""
    name = b"shellsim.abi"
    payload = leb(len(name)) + name + abi
    path.write_bytes(path.read_bytes() + b"\0" + leb(len(payload)) + payload)


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a trusted build command with an explicit optional target environment."""
    subprocess.run(command, cwd=cwd, env=env, check=True)
