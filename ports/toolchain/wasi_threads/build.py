"""Build the SDK tooling inventory and current threaded platform product."""

import json

from ports._support.producer_tools import extract
from ports.api import BuildContext, ProductBuildOutput
from ports.toolchain.wasi_threads.dynamic import build as build_platform
from ports.toolchain.wasi_threads.dynamic import sdk_tooling


def build(ctx: BuildContext) -> ProductBuildOutput:
    recipe = ctx.metadata
    archive = recipe.get("sdk_archive", recipe.get("sdk"))
    if ctx.variant == "tooling":
        root = ctx.work / "sdk"
        ctx.work.mkdir(parents=True)
        extract(ctx.sources["sdk"], root, archive["sha256"], set())
        manifest = ctx.work / "tooling.json"
        manifest.write_text(json.dumps(sdk_tooling(root), sort_keys=True, indent=2) + "\n")
        return ProductBuildOutput(root, manifest)
    seed = ctx.host_seed
    if seed is None or not {"cmake", "ninja"}.issubset(seed.tools):
        raise ValueError("threaded platform requires admitted cmake and ninja")
    root = build_platform(
        ctx.sources["sdk"],
        ctx.sources["wasi_libc"],
        ctx.product_dependencies["compiler"].root,
        seed.tools["cmake"].path,
        seed.tools["ninja"].path,
        ctx.work,
    )
    return ProductBuildOutput(root, root / "manifest.json")
