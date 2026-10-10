"""Normalize pinned OpenBLAS C/Fortran interfaces before a graph build."""

import shutil

from ports._support.native_adapters import NativeBuildContext
from ports.native.openblas.source import prepare_source


def prepare(context: NativeBuildContext) -> None:
    source = context.source
    normalized = source.with_name(source.name + "-normalized")
    prepare_source(source, normalized)
    common = normalized / "common.h"
    text = common.read_text()
    anchor = "#include <time.h>\n#include <math.h>\n#endif"
    if text.count(anchor) != 1:
        raise ValueError("OpenBLAS pthread header anchor must occur exactly once")
    common.write_text(
        text.replace(
            anchor,
            "#include <time.h>\n#include <math.h>\n"
            "#if defined(__wasi__) && defined(USE_LOCKING)\n#include <pthread.h>\n#endif\n#endif",
        )
    )
    shutil.rmtree(source)
    normalized.rename(source)
