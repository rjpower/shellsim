"""Build upstream Regex using the target CPython setuptools overlay."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
