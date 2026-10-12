# Shellsim ports

Ports build pinned upstream packages for Shellsim's WASI guests. Trusted host
workers compile outside the simulator; guest tests run within its resource and
capability boundaries. See [Buildomatic](buildomatic/README.md) for local/Iris
execution, canonical stores, incremental export and the storage proxy.

```sh
uv run --extra buildomatic python -m ports build --all
uv run --extra buildomatic python -m ports export --url-prefix http://127.0.0.1:8080 --simple
uv run --extra buildomatic python -m ports check
```

`--store` accepts a directory or S3/GCS prefix; client state defaults to
`build/ports`. SDK producers and runtime providers are graph dependencies.
`build --all` selects every default recipe; pass explicit roots for a subset or
additional variants. Progress logs go to stderr; command results are JSON on stdout.
Builds emit assets once; export writes a catalog and optional Simple HTML.
`Environment(pypi_packages=["scipy"], pypi_catalog="<catalog URL>")` fetches its
selected wheels, native providers and interpreter without patched uv.

A port's `recipe.json` declares its source pin, exact dependencies, exports and
guest checks. Optional `build.py` exposes `build(ctx: BuildContext)` using
[ports.api](api.py). [Zlib](native/zlib/build.py) is a small native example;
[pure wheel recipes](python/six/recipe.json) need no builder. Declare local
helpers and source files explicitly so workers receive only selected inputs.
Real product variations use finite recipe variants.

Recipes live under `python`, `native` and `toolchain`; `sdks` selects producers
and ABI defaults. Read [native helpers](_support/native_adapters.md),
[Python backends](_support/PYTHON_BACKENDS.md), or [SDK interfaces](_support/SDK.md)
when changing an adapter. Keep package probes in `<port>/tests` and run the
repository gates from [AGENTS.md](../AGENTS.md).
