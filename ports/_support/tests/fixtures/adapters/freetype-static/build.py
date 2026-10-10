"""Exercise the alternate upstream adapter in tests."""

from ports.api import BuildContext, meson


def build(ctx: BuildContext):
    return meson(
        ctx,
        **{
            "configure_args": [
                "--buildtype=release",
                "-Ddefault_library=static",
                "-Dzlib=system",
                "-Dbrotli=disabled",
                "-Dbzip2=disabled",
                "-Dharfbuzz=disabled",
                "-Dmmap=disabled",
                "-Dpng=disabled",
            ],
            "install_prefix": "/usr/local",
            "jobs": 2,
        },
    )
