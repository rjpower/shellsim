"""Build the shared upstream Boost.Python target registry."""

from ports.native.boost_bindings import build_binding


def build(ctx):
    return build_binding(ctx, numpy=False)
