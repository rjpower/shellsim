"""Build pycosat against the admitted SDK and dependencies."""

from ports.api import BuildContext, python_extension


def build(ctx: BuildContext):
    return python_extension(
        ctx,
        module="pycosat",
        sources=["pycosat.c"],
        metadata="PKG-INFO",
        licenses=["LICENSE", "AUTHORS.md"],
        defines=["NGETRUSAGE"],
    )
