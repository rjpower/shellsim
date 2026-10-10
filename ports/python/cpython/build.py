"""Build the threaded runtime and explicit stdlib extension outputs."""

from ports.api import BuildContext, ProductBuildOutput, python_extension


def build(ctx: BuildContext):
    if ctx.variant == "runtime":
        from ports.python.cpython.threaded import build as compile_runtime

        seed = ctx.host_seed
        if seed is None or "make" not in seed.tools or seed.python_helper is None:
            raise ValueError("CPython runtime requires admitted make and a native Python helper")
        products = ctx.product_dependencies
        root = compile_runtime(
            ctx.source,
            seed.python_helper.path,
            products["tooling"].root,
            products["platform"].root,
            products["compiler"].root,
            seed.tools["make"].path,
            ctx.work,
        )
        return ProductBuildOutput(root, root / "manifest.json")
    if ctx.variant == "stdlib-zlib":
        return python_extension(
            ctx,
            module="zlib",
            sources=("Modules/zlibmodule.c",),
            defines=("Py_BUILD_CORE_MODULE",),
            cpython_include_directories=("internal",),
            include_directories=("include",),
            link_inputs=("lib/libz.so",),
            licenses=("LICENSE",),
        )
    if ctx.variant == "stdlib-ctypes":
        return python_extension(
            ctx,
            module="_ctypes",
            sources=tuple(
                "Modules/_ctypes/" + name
                for name in ("_ctypes.c", "callbacks.c", "callproc.c", "stgdict.c", "cfield.c")
            ),
            defines=(
                "dlopen=__wrap_dlopen",
                "dlsym=__wrap_dlsym",
                "dlerror=__wrap_dlerror",
                "dlclose=__wrap_dlclose",
                "HAVE_FFI_CLOSURE_ALLOC=1",
                "HAVE_FFI_PREP_CLOSURE_LOC=1",
                "HAVE_FFI_PREP_CIF_VAR=1",
            ),
            cpython_include_directories=("internal",),
            include_directories=("include",),
            link_inputs=("lib/libffi.so",),
            licenses=("LICENSE",),
        )
    raise ValueError("unsupported CPython output variant")
