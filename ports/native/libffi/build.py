"""Build upstream scalar libffi with the declared Shellsim WASI backend."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import PurePosixPath

from ports._support.native_adapters import NativeBuildCommand, NativeBuildOutput
from ports._support.wasm import mark_abi
from ports.api import BuildContext
from ports.native.dependencies import target_environment


def build(ctx: BuildContext) -> NativeBuildOutput:
    context = ctx.require_native()
    context.build.mkdir(parents=True)
    configuration = context.build / "configure"
    configuration.mkdir()
    environment = target_environment(context.sdk)
    environment.update(
        LC_ALL="C",
        SOURCE_DATE_EPOCH="1756857600",
        PATH=os.pathsep.join(sorted({str(path.parent) for path in context.host_tools.values()})),
        CC=str(context.target_tools["cc"]),
        CFLAGS=shlex.join(context.compiler_flags),
    )
    commands = []

    def run(argv, directory):
        command = NativeBuildCommand(tuple(map(str, argv)), directory)
        commands.append(command)
        with (context.build / f"command-{len(commands)}.log").open("wb") as log:
            subprocess.run(
                command.argv, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    run(
        (
            context.host_tools["sh"],
            context.source / "configure",
            "--host=wasm32-wasip1",
            "--build=x86_64-pc-linux-gnu",
            "--disable-shared",
            "--disable-docs",
            "--prefix=/usr",
        ),
        configuration,
    )
    includes = tuple(
        "-I" + str(path) for path in (configuration / "include", context.source / "include", configuration)
    )
    objects = []
    for name, source in (
        ("prep_cif", context.source / "src/prep_cif.c"),
        ("types", context.source / "src/types.c"),
        ("backend", ctx.port.directory / "shellsim_wasi.c"),
    ):
        output = context.build / (name + ".o")
        run((context.target_tools["cc"], *context.compiler_flags, *includes, "-c", source, "-o", output), context.build)
        objects.append(output)
    prefix = context.staging_prefix / "usr/local"
    for name in ("lib/pkgconfig", "include", "licenses"):
        (prefix / name).mkdir(parents=True)
    library = prefix / "lib/libffi.so"
    run(
        (
            context.target_tools["cc"],
            *context.compiler_flags,
            *context.linker_flags,
            *context.shared_library_flags,
            "-Wl,--export-all,-soname,libffi.so,--fatal-warnings",
            *objects,
            "-o",
            library,
            *context.shared_library_inputs,
        ),
        context.build,
    )
    mark_abi(library, context.abi.encode())
    for name in ("ffi.h", "ffitarget.h"):
        shutil.copyfile(configuration / "include" / name, prefix / "include" / name)
    shutil.copyfile(context.source / "LICENSE", prefix / "licenses/libffi.txt")
    (prefix / "lib/pkgconfig/libffi.pc").write_text(
        "prefix=/usr/local\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
        f"Name: libffi\nDescription: Scalar WASI libffi provider\nVersion: {ctx.port.version}\n"
        "Libs: -L${libdir} -lffi\nCflags: -I${includedir}\n"
    )
    (context.build / "commands.json").write_text(
        json.dumps([{"argv": item.argv, "directory": str(item.directory)} for item in commands], indent=2) + "\n"
    )
    return NativeBuildOutput(context.staging_prefix, tuple(commands), PurePosixPath("/usr/local"))
