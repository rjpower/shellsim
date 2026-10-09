# Unchanged CellPyLib pure wheel

CellPyLib 2.4.0 is published as a source archive only. This approved recipe builds
that pinned archive with pinned setuptools 84.0.0, which includes its wheel build
command. It executes upstream setup.py as build code on the host; it never imports
CellPyLib or executes guest/task code on the host. The artifact records the source,
builder, setuptools wheel, and actual host Python binary/version identities.

```sh
uv run --no-project --python 3.13 python -m ports.cellpylib.build \
  cellpylib-2.4.0.tar.gz setuptools-84.0.0-py3-none-any.whl OUTPUT
```

The backend receives an isolated tool path, sanitized environment and fixed
SOURCE_DATE_EPOCH. The recipe applies no source or import patches. Wheel validation
checks every package file against the source, preserves the upstream license,
checks dependency metadata and pure tags, and verifies all RECORD hashes. Two
fresh builds produced identical wheel bytes in the measured host profile.

Output contains `pure-wheels/cellpylib-2.4.0-py3-none-any.whl` and
`pure-simple/cellpylib/index.html` with a SHA256 link. Merge these into an existing
universe's `pure-wheels` and `pure_index` Simple tree. Its catalog `packages` list
is reserved for native WASI wheels; this pure distribution needs no new registry,
WASI wheel retagging or native ABI manifest.

The unchanged runtime dependencies are numpy>=1.15.4 and matplotlib>=3.0.2;
Requires-Python is >3.6. The initializer eagerly imports Matplotlib plotting,
animation and 3D modules. Building the wheel does not establish guest import
support. The measured private-index install correctly fails resolution because
Matplotlib is absent. A separate diagnostic that stages the unchanged package
payload in real guest CPython also fails at `import matplotlib.pyplot`, with
ModuleNotFoundError. No missing provider is replaced or imported around.
