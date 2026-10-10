"""Stage the declared runtime source component."""

from ports.api import BuildContext, source_tree


def build(ctx: BuildContext):
    return source_tree(ctx)
