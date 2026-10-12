"""Build upstream mmap over the admitted SDK's anonymous-memory emulation."""

from ports.api import BuildContext, python_extension


def build(ctx: BuildContext):
    return python_extension(
        ctx,
        module="mmap",
        sources=("Modules/mmapmodule.c",),
        defines=("Py_BUILD_CORE_MODULE", "_WASI_EMULATED_MMAN=1"),
        cpython_include_directories=("internal",),
        link_inputs=("lib/libwasi-mman.a",),
        licenses=("LICENSE",),
    )
