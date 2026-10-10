"""Stage the patched Wasmtime crate without Cargo packaging metadata."""

from ports.api import BuildContext, source_tree


def build(ctx: BuildContext):
    (ctx.source / "Cargo.toml.orig").unlink()
    return source_tree(ctx)
