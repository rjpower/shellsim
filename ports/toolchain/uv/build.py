"""Build the pinned host resolver as a declared SDK product."""

from ports.api import BuildContext, ProductBuildOutput
from ports.toolchain.uv.producer import build as compile_resolver


def build(ctx: BuildContext) -> ProductBuildOutput:
    binary = compile_resolver(ctx.work, offline=ctx.offline, jobs=ctx.jobs or 2)
    return ProductBuildOutput(binary.parent, binary.parent / "artifact.json")
