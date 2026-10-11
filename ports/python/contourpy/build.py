"""Build upstream contour generators against the target CPython runtime."""

from ports.api import BuildContext, python_meson


def build(ctx: BuildContext):
    return python_meson(
        ctx,
        licenses=["LICENSE"],
        host_header_packages={
            "pybind11": {
                "include": "lib/python3.13/site-packages/pybind11/include",
                "tool": "pybind11-config",
                "version": "3.0.4",
            }
        },
        # Pybind headers contain an unused helper template on this Clang target.
        configure_args=[
            "--buildtype=release",
            "-Dcpp_args=-Wno-error=unused-template",
        ],
    )
