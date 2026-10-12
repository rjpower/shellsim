"""Stage the pinned release IWAD and its notices without a compiler."""

import hashlib
import shutil

from ports.api import BuildContext

WAD_SHA256 = "7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d"


def build(ctx: BuildContext) -> None:
    wad = ctx.source / "freedoom1.wad"
    if hashlib.sha256(wad.read_bytes()).hexdigest() != WAD_SHA256:
        raise ValueError("Freedoom IWAD SHA-256 mismatch")
    for paths in ctx.metadata["exports"].values():
        for path in paths:
            destination = ctx.result / "payload" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ctx.source / destination.name, destination)
