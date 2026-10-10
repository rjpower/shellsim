"""Build freetype against the admitted SDK and dependencies."""

from ports.api import BuildContext, meson


def build(ctx: BuildContext):
    return meson(
        ctx,
        configure_args=[
            "--buildtype=release",
            "-Ddefault_library=shared",
            "-Dzlib=system",
            "-Dbrotli=disabled",
            "-Dbzip2=disabled",
            "-Dharfbuzz=disabled",
            "-Dmmap=disabled",
            "-Dpng=disabled",
        ],
        jobs=2,
        install_prefix="/usr/local",
    )
