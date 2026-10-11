"""Build upstream Shellingham with virtual Linux status support."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
