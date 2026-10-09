"""Public runtime setup and VFS staging for package-owned guest tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shellsim import CPythonRuntime, Environment


@dataclass
class Guest:
    """A fixed interpreter and its isolated, installed package environment."""

    runtime: CPythonRuntime
    environment: Environment
    interpreter: bytes

    def run_script(self, source: Path, *, arguments: Sequence[str] = ()):
        """Stage a checked-in Python probe and execute it only in the guest."""
        path = "/tmp/" + source.name
        self.environment.write_file(path, source.read_bytes())
        return self.runtime.run(self.environment, [path, *arguments])

    def assert_interpreter_unchanged(self) -> None:
        assert self.environment.read_file("/usr/bin/python3.wasm") == self.interpreter


def mount_guest(
    bundle: Path,
    *,
    universe: Path | None = None,
    uv: Path | None = None,
    requirements: Sequence[str] = (),
    wheels: Sequence[Path] = (),
    cpu: int = 10_000_000_000,
    memory: int = 1024**3,
    disk: int = 128 * 1024**2,
) -> Guest:
    """Mount the public CPython runtime and resolve declared package requirements."""
    from shellsim import CPythonRuntime, Environment

    runtime = CPythonRuntime(bundle, universe=universe, uv=uv)
    environment = Environment(cpu=cpu, memory=memory, disk=disk)
    runtime.mount(environment)
    guest = Guest(runtime, environment, environment.read_file("/usr/bin/python3.wasm"))
    if requirements:
        runtime.install_pypi(environment, list(requirements))
    for wheel in wheels:
        runtime.install_wheel(environment, wheel)
    return guest


def stage_script(environment: Environment, source: Path) -> list[str]:
    """Return guest argv after staging a package-owned script in the VFS."""
    path = "/tmp/" + source.name
    environment.write_file(path, source.read_bytes())
    return [path]
