# Pinned Buildomatic task image

This image supplies native tools for one controlled SDK bootstrap and subsequent
Iris port workers. It contains no compiled LLVM product and starts no sccache
daemon. The Iris service owner configures runtime cache settings and credentials.

The accepted image is
`ghcr.io/marin-community/iris-task@sha256:93ac2b06806548e2a03e88f70b240d5625151eaf3851c24ba32fbe6aa4381acc`,
tagged `buildomatic-sdk-v1-20261010-97zs1nvl-r1`. `release.json` records its
registry digest, native seed identity and verification evidence. The earlier
non-`r1` image is unused.

Debian separates GCC programs under `libexec`. The new private package combines
those programs, GCC libraries and complete C++ headers in its recorded support
tree. Fresh `gcc`/`g++` launchers pass an explicit relative `-B` support path and
private C++ include paths. They are constructed before any native receipt is
sealed. No existing script or receipt is rewritten. This layout fits the existing
producer seed interface and remains valid when copied to the stable root.
Pkgconf carries its binary and `libpkgconf.so.3` in a separate complete package;
its new launcher sets an install-relative library path. Ninja is also private.

The base image is pinned by digest in the Dockerfile. Binary inputs are pinned in
`inputs.json`: Kitware CMake 3.31.8 with its complete share tree, Mozilla sccache
0.18.0, a complete native CPython 3.13.7 helper, the existing admitted generator
closure, and exact caller make/shell/rm bytes and aliases. Package installation
supplies GCC/G++, binutils, Ninja, pkgconf and ordinary native build utilities.
The published image digest fixes their actual versions; `preflight.py` records
the package versions and native dynamic-library dependencies.

The stable prepared root is
`/home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1`.
Its partial host seed, generator base/environment, Meson and verified source
archives are copied without changing receipt or launcher bytes. The native
helper and GCC/CMake packages occupy separate image roots under
`/opt/buildomatic`. Complete fresh native receipts are generated later, using the
published image digest. Baking those receipts into the same image would make
their image provenance refer to themselves.

The existing prepared source cache remains the offline producer input. The image
excludes SDK build workspaces, old registries, caller credentials and cache
daemons. Consumers keep `/usr/bin` in PATH through the verified make/sh/rm
bindings. CMake, GCC and Python roots are private. Existing caller system files
are verified in place and are never restored or overwritten.

Build from the repository root with Docker Buildx. Named contexts read the
existing immutable closure directly, avoiding another large host-side copy.
Use task-owned `target/buildomatic-image` for binary downloads and evidence.
Download CMake from the URL in `inputs.json` and verify its SHA256 against
[Kitware's checksum manifest](https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-SHA-256.txt).
Download the pinned pure-Python packaging wheel to the same input directory.
Copy the exact `/usr/bin/make`, `/usr/bin/dash` and `/usr/bin/gnurm` bytes into
`target/buildomatic-image/utilities/{make,dash,gnurm}`. Do not copy credentials.

```sh
docker buildx build --load --progress plain \
  --tag ghcr.io/marin-community/iris-task:buildomatic-sdk-v1-20261010-97zs1nvl-r1 \
  --build-context inputs=target/buildomatic-image/inputs \
  --build-context utilities=target/buildomatic-image/utilities \
  --build-context sccache=/home/power/code/shellsim/.worktrees/buildomatic-core/target/buildomatic-validation/sccache-v0.18.0-x86_64-unknown-linux-musl \
  --build-context helper=/home/power/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu \
  --build-context prepared=/home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1 \
  --build-context code=ports \
  ports/buildomatic/images
```

The Dockerfile runs the native preflight before completing the image. Repeat it
in a cheap container before pushing:

```sh
docker run --rm --network none --cpus 1 --memory 2g \
  ghcr.io/marin-community/iris-task:buildomatic-sdk-v1-20261010-97zs1nvl-r1 \
  python3 -B /opt/buildomatic/preflight.py \
  > target/buildomatic-image/container-preflight.json
```

The preflight re-admits the exact generator seed, verifies every cached source
and helper file, checks utility bytes/aliases, loads Python extensions, checks
GCC's private `cc1`/`cc1plus`, queries CMake `FindThreads`, and verifies sccache.
It does not compile a source file. Verify copied GCC/CMake tools
also run on the caller before claiming compatible image/caller execution.
The recorded acceptance additionally configures a tiny CMake project with
`find_package(Threads REQUIRED)`, then builds and runs a C++ thread probe on both
the caller and image using the final stable GCC/CMake/Ninja paths. This compiles
only the probe; no SDK source or LLVM product is built.

Push only after these checks, then record the immutable registry digest and
preflight evidence. Use that digest for remote tasks. Stage the fresh native host
seed with full packages, rather than copying their executables alone:

```sh
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1,dst=/home/power/code/shellsim/.worktrees/shellsim-ports-rollout/target/buildomatic-sdk-bootstrap-v1 \
  --workdir /opt/buildomatic \
  ghcr.io/marin-community/iris-task@sha256:93ac2b06806548e2a03e88f70b240d5625151eaf3851c24ba32fbe6aa4381acc \
  python3 -B -m ports.buildomatic.images.stage_native \
  --image ghcr.io/marin-community/iris-task@sha256:93ac2b06806548e2a03e88f70b240d5625151eaf3851c24ba32fbe6aa4381acc
```

This creates immutable receipts and `host-seed.ready.json`, without altering
the prepared seed or existing receipts. New native receipts occupy
`host/native-receipts`, separate from the already baked generator receipts, so
workers can restore complete missing roots without adding files to an existing
immutable mount. It rejects existing native output rather
than overwriting. Then handle the independent resolver product. LLVM bootstrap
still requires actual remote-port acceptance and the
root coordinator's launch instruction; image publication grants neither.

Native staging has completed at the stable root. Its admitted
`host-seed.ready.json` has SHA256
`7f10db9fee947ac0a32241430e694b9d3f695ddb2471c366a764bcb7d9bbddd1`.
The original prepared seed remains unchanged. Repeating the staging command
rejects the existing native output. Consumers use the ready seed and restore
missing native roots through the portable descriptor's explicit original root
bindings.
