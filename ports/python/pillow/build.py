"""Build pillow against the admitted SDK and dependencies."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(
        ctx,
        config_settings={
            "avif": "disable",
            "freetype": "enable",
            "imagequant": "disable",
            "jpeg": "enable",
            "jpeg2000": "disable",
            "lcms": "disable",
            "parallel": "2",
            "platform-guessing": "disable",
            "raqm": "disable",
            "tiff": "disable",
            "webp": "disable",
            "xcb": "disable",
            "zlib": "enable",
        },
    )
