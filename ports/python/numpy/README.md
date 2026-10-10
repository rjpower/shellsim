# NumPy

NumPy 2.3.5 builds through upstream Meson with typed Python options. The SDK supplies the target Python headers, generated configuration, compiler and resources. Threading is enabled; optional CPU dispatch and host-specific acceleration are disabled. Multiple extension modules retain upstream package layout and metadata.

`recipe.json` owns source pins, exact graph dependencies, exports and checks.
`build.py` exposes the typed `build(ctx)` entrypoint and package-specific options.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports python/numpy --store /path/to/ports-store \
  --output /path/to/release --check
```

See the [ports authoring guide](../../README.md) for SDK materialization,
retained workspaces and cache admission. Historical sealed artifact receipts retain
their original identities; obsolete production build CLIs are retired.
