# SDK products

The default `wasi-threads-v3` SDK declares LLVM, threaded libc, archive tooling
and CPython producers. The planner adds the required producers and runtime
providers to the same graph as packages. Workers use installed host tools and
fetch missing products from the canonical store; the coordinator needs no SDK.
Local and Iris execution use the same worker.

```sh
uv run --extra buildomatic python -m ports build native/freetype python/kiwisolver --jobs 64
uv run --extra buildomatic python -m ports check
uv run --extra buildomatic python -m ports export --url-prefix http://127.0.0.1:8080
```

Use `--store <directory-or-cloud-prefix>` for products and `--state <directory>`
for graph/results metadata. Source pins, producer code and declared dependencies
select cached products. Complete worker cache entries are reused without tree
scans. LLVM retains compatible Ninja state; workers enable installed sccache.
The catalog installer uses ordinary uv and needs no patched resolver.
