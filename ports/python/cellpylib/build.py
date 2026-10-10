"""Build the upstream pure CellPyLib wheel with the pinned offline backend."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx, preserve_package="cellpylib", preserve_license="LICENSE.txt")
