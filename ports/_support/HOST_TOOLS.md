# Scientific host tools

`host-tools.json` pins the consumed Meson source and generator wheels.
Meson 1.8.3 comes from NumPy 2.3.5's `vendored-meson/meson` directory.
That tree includes NumPy's BLAS and Python integration changes; it differs from
unmodified upstream Meson 1.8.3. The producer applies the separately pinned
WASI archive-group patch and checks its exact input and output bytes.

Host receipts include every regular environment file and each internal alias.
Production can copy the explicit base and environments into a private read-only
tree. Other sessions using the shared uv interpreter cannot change that closure.
Python-backed tools also bind the explicit base Python 3.13.15 executable,
standard library, extension modules and internal aliases. Admission rejects
changed code, missing receipts, an unpinned source definition, a different base
interpreter, a writable base tree, or system site packages. Version verification
reads the bound Python header; admission never executes an interpreter. No
ambient Python discovery is used.

Set the repository cache variables and prevent bytecode writes when creating or
using these environments:

```sh
export UV_CACHE_DIR=/home/power/.cache/uv
export CARGO_TARGET_DIR=/home/power/.cache/cargo-build
export PYTHONDONTWRITEBYTECODE=1
```

For a fresh environment, supply an already admitted base Python explicitly:

```sh
uv venv --copies --python "$BASE_PYTHON/bin/python3.13" "$HOST_ENV"
uv run --no-project --python "$BASE_PYTHON/bin/python3.13" python -B - "$ADMITTED_UV" "$HOST_ENV" "$SOURCE_CACHE" <<'PY'
import subprocess
import sys
from pathlib import Path
from ports._support.host_tools import definition
from ports._support.store import fetch

uv, environment, cache = map(Path, sys.argv[1:])
wheels = [fetch(source, cache) for source in definition()['python']['packages'].values()]
subprocess.run([uv, 'pip', 'install', '--no-index', '--no-deps', '--python',
                environment / 'bin/python', *wheels], check=True)
PY
```

Then produce a fresh Meson tree and receipts. Supply a native host seed with
the remaining host bindings; the producer writes a new seed without target
products:

```sh
uv run --no-project --python "$HOST_ENV/bin/python" python -B -m ports._support.host_tools \
  --host-seed "$INPUT_HOST_SEED" --output "$NEW_RECEIPTS" --cache "$SOURCE_CACHE" \
  --base-python "$BASE_PYTHON" --python-environment "$HOST_ENV" \
  --meson-root "$NEW_MESON" --meson-python-environment "$HOST_ENV" \
  --private-python "$PRIVATE_PYTHON" --materialize-meson
```

`--private-python` creates fresh read-only base and environment copies, updates
`pyvenv.cfg` and script paths, and uses those copies in the new descriptor.
Existing destinations are rejected. Omit this option only when the supplied
base is already a private read-only tree.

Omit `--materialize-meson` to verify an existing Meson installation against the
same pinned source and patch. `--offline` requires all pinned archives in the
source cache. Both modes check installed wheel files against the upstream
archives, write new receipts, and fully admit the resulting `host-seed.json`.
Existing output directories are rejected. A failed proof does not authorize
rewriting an old receipt or changing a retained numerical build tree.
