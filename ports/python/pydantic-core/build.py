"""Build the upstream Rust validator and preserve its Python package tree."""

from ports._support.python_rust import build_rust_extension
from ports.api import BuildContext


def build(ctx: BuildContext):
    return build_rust_extension(
        ctx,
        module="pydantic_core._pydantic_core",
        python_sources=(("python/pydantic_core", "pydantic_core"),),
        licenses=("LICENSE",),
    )
