"""Build openblas against the admitted SDK and dependencies."""

from ports.api import BuildContext, plain_make
from ports.native.openblas.graph_install import install
from ports.native.openblas.graph_prepare import prepare


def build(ctx: BuildContext):
    context = ctx.require_native()
    prepare(context)
    output = plain_make(
        ctx,
        jobs=2,
        build_targets=["libs", "netlib"],
        install_targets=[],
        build_args=[
            "NOFORTRAN=1",
            "NO_LAPACKE=1",
            "USE_THREAD=0",
            "USE_LOCKING=1",
            "NUM_THREADS=1",
            "TARGET=RISCV64_GENERIC",
            "BINARY=32",
            "ARCH=wasm32",
            "OSNAME=EMBEDDED",
        ],
    )
    install(context, ctx.port.version)
    return output
