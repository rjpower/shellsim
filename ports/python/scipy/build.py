"""Build scipy against the admitted SDK and dependencies."""

from ports.api import BuildContext, python_meson


def build(ctx: BuildContext):
    return python_meson(
        ctx,
        jobs=8,
        metadata="PKG-INFO",
        licenses=["LICENSE.txt"],
        dependency_properties={"numpy-include-dir": {"path": "python/numpy/_core/include", "port": "python/numpy"}},
        host_header_packages={
            "pybind11": {
                "include": "lib/python3.13/site-packages/pybind11/include",
                "tool": "pybind11-config",
                "version": "3.0.4",
            }
        },
        configure_args=[
            "--buildtype=release",
            "-D_without-fortran=true",
            "-Duse-pythran=false",
            "-Duse-g77-abi=true",
            "-Dblas=openblas",
            "-Dlapack=openblas",
            "-Dcpp_args=-D_WASI_EMULATED_SIGNAL",
        ],
    )
