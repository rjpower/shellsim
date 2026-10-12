# Buildomatic worker image

Ubuntu supplies ordinary build tools; uv supplies Python generators and Shellsim
for guest checks. SDKs and products stay in the canonical store. Build the host
wheel in Ubuntu 24.04 with Cargo/uv caches mounted, then use cached Docker layers.
Run the first command inside that environment:

```sh
uv run --with maturin maturin build --profile dev --strip --out ports/buildomatic/images/wheels
docker buildx build --push --platform linux/amd64 \
  --tag ghcr.io/marin-community/iris-task:buildomatic-unified-v8 ports/buildomatic/images
```

Pin the returned digest in `IrisConfig`. Its `setup_scripts` should install the
wheel into Iris's Python environment after `uv sync`:
`uv pip install /opt/buildomatic/wheels/*.whl`. Enable `compiler_cache` for regional
sccache. Allocate CPUs, RAM and disk for the selected producers at BATCH priority;
`--jobs` controls compilation within each port.
