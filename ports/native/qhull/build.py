"""Build only the reentrant Qhull library consumed by Matplotlib."""

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    return cmake(
        ctx,
        configure_args=[
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_STATIC_LIBS=OFF",
            "-DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON",
        ],
        build_targets=["qhull_r"],
    )
