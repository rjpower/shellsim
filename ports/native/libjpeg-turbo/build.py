"""Build libjpeg-turbo against the admitted SDK and dependencies."""

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    return cmake(
        ctx,
        configure_args=[
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD=2.1.5.1",
            "-DENABLE_SHARED=ON",
            "-DENABLE_STATIC=OFF",
            "-DWITH_SIMD=OFF",
            "-DWITH_TURBOJPEG=OFF",
            "-DWITH_JAVA=OFF",
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-Bdynamic",
        ],
        jobs=2,
        install_prefix="/usr/local",
        executable_sdk_link_inputs=["lib/wasm32-wasip1-threads/libsetjmp.a"],
    )
