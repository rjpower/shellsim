"""Build kiwisolver against the admitted SDK and dependencies."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
