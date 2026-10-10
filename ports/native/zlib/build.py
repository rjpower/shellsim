"""Build zlib against the admitted SDK and dependencies."""

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    return cmake(
        ctx,
        configure_args=["-DZLIB_BUILD_EXAMPLES=OFF"],
        build_targets=["zlib", "zlibstatic"],
        install_targets=["install"],
        jobs=2,
        install_prefix="/usr/local",
    )
