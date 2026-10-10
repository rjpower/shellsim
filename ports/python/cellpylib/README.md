# CellPyLib

CellPyLib 2.4.0 builds through the shared offline setuptools backend. The helper preserves the admitted package source and LICENSE.txt unchanged and seals deterministic RECORD hashes. Its upstream dependency constraints include NumPy and matplotlib. This tree supplies NumPy; matplotlib is not a curated native port, so this recipe does not establish a complete guest matplotlib installation.

`recipe.json` owns source pins, exact graph dependencies, exports and checks.
`build.py` exposes the typed `build(ctx)` entrypoint and package-specific options.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports python/cellpylib --store /path/to/ports-store \
  --output /path/to/release
```

See the [ports authoring guide](../../README.md) for SDK materialization,
retained workspaces and cache admission. Historical sealed artifact receipts retain
their original identities; obsolete production build CLIs are retired.
