"""Normalize pinned OpenBLAS C/Fortran interfaces before a graph build."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import check_build_scripts
from ports.native.openblas.build import prepare_source


def main():
    directory = Path(__file__).parent
    check_build_scripts(json.loads((directory / "graph-recipe.json").read_text()), directory)
    context = json.loads(Path(sys.argv[1]).read_text())
    source = Path(context["source"])
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


if __name__ == "__main__":
    main()
