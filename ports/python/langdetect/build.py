"""Build the pinned upstream Python project with target setuptools."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
