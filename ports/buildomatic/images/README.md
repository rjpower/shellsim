# Buildomatic worker image

Ubuntu supplies ordinary build tools; uv supplies Python generators and Shellsim
for guest checks. SDKs and products stay in the canonical store. Build from the
repository root; the image builds its Shellsim wheel with cached Cargo and uv inputs.

```sh
docker buildx build --push --platform linux/amd64 \
  --tag ghcr.io/marin-community/iris-task:buildomatic \
  --file ports/buildomatic/images/Dockerfile .
```

Pin the returned digest in `IrisConfig`. Its `setup_scripts` should install the
wheel into Iris's Python environment after `uv sync`:
`uv pip install /opt/buildomatic/wheels/*.whl`. Enable `compiler_cache` for regional
sccache. Allocate CPUs, RAM and disk for the selected producers at BATCH priority;
`--jobs` controls compilation within each port.
