# Buildomatic worker image

The worker uses ordinary Ubuntu packages for GCC, headers, CMake, Ninja,
pkg-config and other build utilities. uv installs a Python 3.13 build helper
and Python generators in an image environment. sccache is installed as a
normal executable. Rustup installs the declared Rust toolchain for the patched
host resolver. Iris transports its runtime and the build code separately.

No SDK, caller worktree, private host-tool inventory or build output is baked
into this image. SDKs and port products are fetched from the configured store
and retained in the worker's artifact cache. Docker layers cache dependency
installation across code changes.

Build and publish from the repository root:

```sh
docker buildx build --push --platform linux/amd64 \
  --tag ghcr.io/marin-community/iris-task:buildomatic-ubuntu-v4 \
  ports/buildomatic/images
```

Use the returned image digest for an Iris task. The service config enables
sccache and supplies the regional storage environment; credentials remain in
Iris. Match make/Ninja jobs to the task's CPU allocation. Start SDK bootstrap
with 128 CPUs, 192 GiB RAM and 200 GiB disk at BATCH priority.
