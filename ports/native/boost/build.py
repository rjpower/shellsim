"""Export upstream Boost headers for the first native chemistry build."""

import shutil
from pathlib import PurePosixPath

from ports._support.native_adapters import NativeBuildOutput
from ports.api import BuildContext


def build(ctx: BuildContext) -> NativeBuildOutput:
    native = ctx.require_native()
    prefix = native.staging_prefix / "usr/local"
    shutil.copytree(native.source / "boost", prefix / "include/boost")
    return NativeBuildOutput(native.staging_prefix, (), PurePosixPath("/usr/local"))
