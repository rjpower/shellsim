# Buildomatic

Buildomatic runs trusted host builds outside Shellsim guests. Local builds use a
directory; Iris workers fetch components from the canonical S3 cache on demand.
Builds emit wheels, native providers, and a separate interpreter asset once;
export reads `graph.json` and writes catalog metadata, with optional Simple HTML.
State defaults to `build/ports`.

```sh
uv run --no-project python -m ports build python/scipy --backend iris --jobs 128
uv run --no-project python -m ports export --url-prefix http://127.0.0.1:8080 --simple
uv run python ports/buildomatic/s3_proxy.py --prefix s3://my-bucket/ports
```

Omit `--backend iris` for local builds. Iris uses `build/ports/service.json`;
bootstrap its SDK explicitly with `ports.buildomatic.bootstrap --python` first.
`--max-workers` controls port concurrency; `--jobs` controls compilation within
one port. Export to another store with `--output <prefix> --copy-assets` to copy
only missing assets. Cloud access needs `marin-rigging` bucket settings.

Point `Environment(pypi_packages=["scipy"], pypi_catalog="<catalog.json URL>")`
at the exported catalog. It resolves the curated WASI closure and caches only
selected assets, without patched uv or SDK downloads. The older cohort workflow
in `ports.buildomatic.publish` still requires its patched resolver.
