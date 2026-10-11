"""Compile the upstream bzip2 library with the selected WASI toolchain."""

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    # Upstream supplies Makefiles; CMake shares the normal SDK and cache setup.
    (ctx.source / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.16)
project(bzip2 C)
set(sources blocksort.c huffman.c crctable.c randtable.c compress.c decompress.c bzlib.c)
add_library(bz2 SHARED ${sources})
add_library(bz2_static STATIC ${sources})
set_target_properties(bz2_static PROPERTIES OUTPUT_NAME bz2)
install(TARGETS bz2 bz2_static LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES bzlib.h DESTINATION include)
"""
    )
    return cmake(ctx, jobs=32)
