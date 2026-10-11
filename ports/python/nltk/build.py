"""Build upstream NLTK with an explicit WASI download frontier."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
