"""Build upstream extensions using target NumPy headers and host generators."""

from ports.api import BuildContext, python_meson


def build(ctx: BuildContext):
    return python_meson(
        ctx,
        licenses=["LICENSE"],
        dependency_properties={"numpy-include-dir": {"path": "python/numpy/_core/include", "port": "python/numpy"}},
        configure_args=["--buildtype=release", "-Db_lto=false"],
    )
