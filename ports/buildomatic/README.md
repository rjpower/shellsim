# Buildomatic

`ports build`, `ports export` and `ports check` share a graph and worker. Local
execution runs that worker here; Iris dispatches it to a pool. `--store` selects
a canonical directory or S3/GCS prefix independently of execution. Client state
(`graph.json`, `results.json`) defaults to `build/ports`. Iris requires Python 3.12 or 3.13.

```sh
uv run --extra buildomatic python -m ports build --all --jobs 128
uv run --extra buildomatic python -m ports export --url-prefix http://127.0.0.1:8080 --simple
uv run --extra buildomatic python -m ports check
uv run --extra buildomatic python -m ports build --all --backend iris --max-workers 4 --store s3://bucket/ports
```

`--all` selects every default recipe; export/check reuse the saved roots.
Iris starts on `cw-us-east-02a` and stops its owned pool on exit. Use
`--worker-cpus`, `--worker-memory-gb`, `--worker-disk-gb` and `--iris-image` to size it;
defaults are 64 CPUs, 32 GiB RAM and 100 GiB disk per worker, with sccache enabled.
`--iris-service service.json` connects to an existing pool and leaves it running.
SDK producers are graph dependencies. `--max-workers` controls port concurrency; `--jobs` controls
compilation within a port. The trusted local supervisor checks resident RAM
periodically; Iris also applies its container limits. Builds emit assets once; export can promote missing
assets with `--output <prefix> --copy-assets`. Serve the store with
`s3_proxy.py --prefix <prefix>` and pass the catalog URL to
`Environment(pypi_packages=["scipy"], pypi_catalog="<URL>")`; no patched uv is needed.
