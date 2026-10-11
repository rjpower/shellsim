"""Build the isolated scheduler translator as a cached host dependency."""

import shutil
from pathlib import Path

from ports._support.rust_tools import run_cargo, rust_environment, rust_tools
from ports.api import BuildContext


def build(ctx: BuildContext) -> None:
    environment = rust_environment(ctx)
    _, cargo, _, _ = rust_tools("1.97.1", environment)
    command = [str(cargo), "build", "--release", "--locked", "--message-format=json"]
    if ctx.offline:
        command.append("--offline")
    run_cargo(command, ctx.source, environment, ctx.result.parent / "scheduler-cargo.log")
    binary = Path(environment["CARGO_TARGET_DIR"]) / "release/shellsim-wasm-scheduler"
    destination = ctx.result / "bin/shellsim-wasm-scheduler"
    destination.parent.mkdir()
    shutil.copyfile(binary, destination)
    destination.chmod(0o755)
