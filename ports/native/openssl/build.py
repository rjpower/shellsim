"""Build upstream OpenSSL archives with cached WASI C compilation."""

import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from ports._support.native_adapters import (
    NativeBuildCommand,
    NativeBuildOutput,
    build_environment,
    compiler_wrapper_text,
)
from ports.api import BuildContext


def build(ctx: BuildContext) -> NativeBuildOutput:
    native = ctx.require_native()
    native.build.mkdir(parents=True)
    tools = native.build / "adapter-tools"
    tools.mkdir()
    response = Path(__file__).parents[2] / "_support/compiler_response.py"
    for role in ("cc", "cxx"):
        wrapper = tools / role
        wrapper.write_text(compiler_wrapper_text(native, role, response.read_text()))
        wrapper.chmod(0o755)
    environment = build_environment(native, {})
    perl = shutil.which("perl", path=environment["PATH"])
    if perl is None:
        raise FileNotFoundError("OpenSSL needs the ordinary host Perl interpreter")
    environment["PERL"] = perl
    commands = []

    def run(argv):
        command = NativeBuildCommand(tuple(map(str, argv)), native.build)
        commands.append(command)
        log = native.build / f"command-{len(commands)}.log"
        with log.open("wb") as output:
            result = subprocess.run(
                command.argv, cwd=native.build, env=environment, stdout=output, stderr=subprocess.STDOUT
            )
        if result.returncode:
            error = subprocess.CalledProcessError(result.returncode, command.argv)
            with log.open("rb") as output:
                output.seek(max(0, log.stat().st_size - 8192))
                error.add_note(output.read().decode(errors="replace"))
            raise error

    run(
        (
            perl,
            native.source / "Configure",
            "gcc",
            "--prefix=/usr/local",
            "--libdir=lib",
            "no-shared",
            "no-asm",
            "no-threads",
            "no-apps",
            "no-tests",
            "no-sock",
            "no-dso",
            "no-engine",
            "no-ui-console",
            "no-async",
            "no-secure-memory",
            "--with-rand-seed=getrandom",
            "-fPIC",
        )
    )
    run((native.host_tools["make"], "-s", f"-j{ctx.jobs or os.cpu_count() or 1}", "build_libs"))
    run((native.host_tools["make"], f"DESTDIR={native.staging_prefix}", "install_dev"))
    return NativeBuildOutput(native.staging_prefix, tuple(commands), PurePosixPath("/usr/local"))
