"""Build numpy against the admitted SDK and dependencies."""

from ports.api import BuildContext, python_meson


def build(ctx: BuildContext):
    return python_meson(
        ctx,
        jobs=8,
        metadata="PKG-INFO",
        licenses=["LICENSE.txt", "LICENSES_bundled.txt"],
        cross_properties={"longdouble_format": "IEEE_QUAD_LE"},
        configure_args=[
            "--buildtype=release",
            "-Dblas=none",
            "-Dlapack=none",
            "-Ddisable-threading=false",
            "-Ddisable-optimization=true",
            "-Ddisable-highway=true",
            "-Ddisable-intel-sort=true",
            "-Dcpu-baseline=none",
            "-Dcpu-dispatch=none",
        ],
    )
