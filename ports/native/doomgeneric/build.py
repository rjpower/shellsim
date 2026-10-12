"""Compile the pinned Doom engine once using the admitted host SDK."""

import re

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    """Use the upstream source list while replacing its host platform backend."""
    source = ctx.source / "doomgeneric"
    objects = next(
        line.removeprefix("SRC_DOOM = ").split()
        for line in (source / "Makefile.soso").read_text().splitlines()
        if line.startswith("SRC_DOOM = ")
    )
    objects.remove("doomgeneric_soso.o")
    if not objects or any(re.fullmatch(r"[A-Za-z0-9_-]+\.o", name) is None for name in objects):
        raise ValueError("invalid Doomgeneric source list")
    sources = "\n".join("doomgeneric/" + name[:-2] + ".c" for name in objects)
    (ctx.source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(doomgeneric C)\n"
        "add_executable(doom\n" + sources + "\ndoomgeneric/doomgeneric_shellsim.c)\n"
        "target_include_directories(doom PRIVATE doomgeneric)\n"
        "target_compile_definitions(doom PRIVATE SHELLSIM_WASI=1 _DEFAULT_SOURCE=1)\n"
        "target_compile_options(doom PRIVATE -O2 -g0)\n"
        "install(TARGETS doom RUNTIME DESTINATION bin)\n"
    )
    return cmake(ctx, jobs=ctx.jobs or 2)
