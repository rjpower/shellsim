"""Build upstream Redis with its pinned Hatchling backend."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
