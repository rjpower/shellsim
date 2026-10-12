"""Build only upstream WASI memory-map emulation as a PIC target archive."""

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    (ctx.source / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.16)
project(wasi_mman C)
add_library(wasi-mman STATIC libc-bottom-half/mman/mman.c)
set_target_properties(wasi-mman PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS wasi-mman ARCHIVE DESTINATION lib)
"""
    )
    return cmake(ctx, jobs=1)
