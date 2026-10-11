"""Build Matplotlib with the declared FreeType and reentrant Qhull ports."""

from ports.api import BuildContext, python_meson


def build(ctx: BuildContext):
    return python_meson(
        ctx,
        licenses=["LICENSE/LICENSE"],
        host_header_packages={
            "pybind11": {
                "include": "lib/python3.13/site-packages/pybind11/include",
                "tool": "pybind11-config",
                "version": "3.0.4",
            }
        },
        configure_args=[
            "--buildtype=release",
            "-Dsystem-freetype=true",
            "-Dsystem-qhull=true",
            "-Dmacosx=false",
            "-DrcParams-backend=Agg",
            "-Db_lto=false",
        ],
        install_tags=["data", "python-runtime", "runtime"],
    )
