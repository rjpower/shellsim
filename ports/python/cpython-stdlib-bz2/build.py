"""Build upstream CPython's bz2 extension without relinking the interpreter."""

from ports.api import BuildContext, python_extension


def build(ctx: BuildContext):
    return python_extension(
        ctx,
        module="_bz2",
        sources=("Modules/_bz2module.c",),
        defines=("Py_BUILD_CORE_MODULE",),
        cpython_include_directories=("internal",),
        include_directories=("include",),
        link_inputs=("lib/libbz2.so",),
        licenses=("LICENSE",),
    )
