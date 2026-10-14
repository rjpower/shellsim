# Buildomatic

`ports build`, `ports export` and `ports check` share a graph and worker. Local
execution runs that worker here; Iris dispatches it to a pool. `--store` selects
a canonical directory or S3/GCS prefix independently of execution. Client state
(`graph.json`, `results.json`) defaults to `build/ports`. Iris requires Python 3.12 or 3.13.
`cache.py` owns product identities and completion records; packaging revisions affect derived
assets without invalidating raw builds. Existing records retain their asset URLs.

```sh
uv run --extra buildomatic python -m ports build --all --jobs 128
uv run --extra buildomatic python -m ports export --url-prefix http://127.0.0.1:8080 --simple
uv run --extra buildomatic python -m ports check
uv run --extra buildomatic python -m ports build --all --backend iris --max-workers 4 --store s3://bucket/ports
```

`--all` selects every guest recipe variant; export/check reuse the saved roots.
Iris starts on `cw-us-east-02a` and stops its owned pool on exit. Use
`--worker-cpus`, `--worker-memory-gb`, `--worker-disk-gb` and `--iris-image` to size it;
defaults are 64 CPUs, 32 GiB RAM and 100 GiB disk per worker, with sccache enabled.
`--iris-service service.json` connects to an existing pool and leaves it running.
SDK producers are graph dependencies. `--max-workers` controls port concurrency; `--jobs` controls
compilation within a port. The trusted local supervisor checks resident RAM
periodically; Iris also applies its container limits. Builds emit assets once; export can promote missing
assets with `--output <prefix> --copy-assets`; `--asset-output <shared-prefix>` separates snapshot
catalogs from shared assets. Serve the store with `s3_proxy.py --prefix <prefix>` and pass the catalog URL to
`Context(pypi_catalog="<URL>").Environment(pypi_packages=["scipy"])`; no patched uv is needed.

Delivery publishes portable wheels and runtime archives unchanged, with separate
Wasmtime companions for the runtime, wheel extensions and native exports. Each
`companions` entry identifies its host `target`, runtime `build_id` and asset path;
wheel entries also name their source member. Delivery keys include that target and ABI,
while portable asset paths remain independent of both. Workers compile baseline code for
their own host triple and must match the planner's target and ABI.
Set `trusted_precompiled=True` on `Context` only for trusted catalog and producer bytes:
companions contain native host code. Setup downloads matching variants for the selected
packages; they load on first use, with raw Wasm fallback, and never enter the guest VFS.
Python installs omit development bundles; explicitly selected tools include them.
