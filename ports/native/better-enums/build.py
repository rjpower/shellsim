"""Export the header dependency pinned by upstream RDKit."""

import shutil
from pathlib import PurePosixPath

from ports._support.native_adapters import NativeBuildOutput
from ports.api import BuildContext


def build(ctx: BuildContext) -> NativeBuildOutput:
    native = ctx.require_native()
    include = native.staging_prefix / "usr/local/include"
    include.mkdir(parents=True)
    shutil.copyfile(native.source / "enum.h", include / "enum.h")
    return NativeBuildOutput(native.staging_prefix, (), PurePosixPath("/usr/local"))
