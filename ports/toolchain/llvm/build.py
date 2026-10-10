"""Build explicit host compiler, guest compiler and guest development products."""

from ports.api import BuildContext, ProductBuildOutput


def build(ctx: BuildContext):
    if ctx.variant == "host":
        from ports.toolchain.llvm.compiler import build as compile_host

        seed = ctx.host_seed
        if seed is None or set(seed.compiler_tools) != {"cc", "cxx", "cmake", "ninja"}:
            raise ValueError("host compiler requires admitted cc, cxx, cmake and ninja")
        root = compile_host(
            ctx.source, *(seed.compiler_tools[name].path for name in ("cc", "cxx", "cmake", "ninja")), ctx.work
        )
        return ProductBuildOutput(root, root / "manifest.json")
    if ctx.variant == "guest":
        from ports.toolchain.llvm.guest import build_guest

        return build_guest(ctx.require_native(), ctx.metadata, ctx.sdk.llvm, jobs=ctx.jobs)
    if ctx.variant == "development":
        from ports.toolchain.llvm.guest import install_guest_sdk

        return install_guest_sdk(ctx.require_native(), ctx.metadata)
    raise ValueError("unsupported LLVM output variant")
