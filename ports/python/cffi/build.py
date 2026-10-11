"""Build upstream CFFI against the target libffi provider."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
