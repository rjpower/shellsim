# CellPyLib

CellPyLib 2.4.0 builds through the shared offline setuptools backend. The helper preserves the admitted package source and LICENSE.txt unchanged and seals deterministic RECORD hashes. Its runtime graph supplies NumPy and Matplotlib. The declared guest probe evolves a fixed Rule 30 automaton, renders a PNG with Agg, and rejects an invalid initialization size.

`recipe.json` owns source pins, exact graph dependencies, exports and checks.
`build.py` exposes the typed `build(ctx)` entrypoint and package-specific options.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports build python/cellpylib --store /path/to/ports-store \
  --state build/ports
```

See the [ports authoring guide](../../README.md) for SDK materialization,
retained workspaces and cache admission. Historical sealed artifact receipts retain
their original identities; obsolete production build CLIs are retired.
