"""Use ordinary host Rust and shared caches without fingerprinting executables."""

import os
import subprocess
from pathlib import Path

from ports._support.build_context import BuildContext


def rust_environment(ctx: BuildContext) -> dict[str, str]:
    """Keep configured Cargo caches and enable the selected compiler launcher."""
    environment = dict(os.environ)
    environment.setdefault("CARGO_TARGET_DIR", str(Path.home() / ".cache/cargo-build"))
    environment["CARGO_BUILD_JOBS"] = str(ctx.jobs or os.cpu_count() or 1)
    if ctx.compiler_cache is not None:
        environment.update(dict(ctx.compiler_cache.environment))
        environment["RUSTC_WRAPPER"] = str(ctx.compiler_cache.path)
    return environment


def rust_tools(version: str, environment: dict[str, str]) -> tuple[Path, Path, Path, str]:
    """Locate the declared ordinary toolchain and its host libraries."""

    def locate(name: str) -> Path:
        return Path(
            subprocess.check_output(
                ["rustup", "which", "--toolchain", version, name], env=environment, text=True
            ).strip()
        )

    rustc, cargo = locate("rustc"), locate("cargo")
    sysroot = Path(subprocess.check_output([str(rustc), "--print", "sysroot"], env=environment, text=True).strip())
    details = subprocess.check_output([str(rustc), "--version", "--verbose"], env=environment, text=True)
    host = next(line.removeprefix("host: ") for line in details.splitlines() if line.startswith("host: "))
    return rustc, cargo, sysroot, host


def run_cargo(argv: list[str], source: Path, environment: dict[str, str], log: Path) -> subprocess.CompletedProcess:
    """Retain useful Cargo diagnostics while returning its artifact records."""
    result = subprocess.run(
        argv, cwd=source, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        error = subprocess.CalledProcessError(result.returncode, argv)
        error.add_note(result.stderr[-8192:])
        raise error
    return result
