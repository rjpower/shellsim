"""Build upstream persistent containers with the cached Rust target libraries."""

from ports._support.python_rust import build_rust_extension
from ports.api import BuildContext


def build(ctx: BuildContext):
    return build_rust_extension(ctx, module="rpds", licenses=("LICENSE",), stubs=("rpds.pyi",))
