"""Build numpy against the admitted SDK and dependencies."""

from dataclasses import replace

from ports.api import BuildContext, python_meson


def build(ctx: BuildContext):
    # NumPy's features module is supplied by its vendored Meson, not PyPI Meson.
    native = ctx.require_native()
    meson = ctx.source / "vendored-meson/meson/meson.py"
    meson.chmod(0o755)
    ctx = replace(ctx, native=replace(native, host_tools={**native.host_tools, "meson": meson}))
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
