"""Build upstream NumPy converters against the shared Boost.Python registry."""

from ports.native.boost_bindings import build_binding


def build(ctx):
    return build_binding(ctx, numpy=True)
