# Buildomatic

Buildomatic runs trusted host builds, isolated from Shellsim guests. Local builds
use a directory; Iris workers fetch SDK components and dependencies from their
canonical S3 cache on demand. The client reads metadata, without downloading an SDK.

```sh
uv run --no-project python -m ports build python/six
uv run --no-project python -m ports export --url-prefix http://127.0.0.1:8080 --simple
```

State defaults to `build/ports`, including `graph.json`, `results.json` and
`export/catalog.json`. Builds emit individual wheels/providers once. Export reads
the saved graph; `--simple` adds repository HTML. The catalog installer is a follow-up.
For another publication store, add `--output <directory-or-cloud-prefix> --copy-assets`;
only missing assets are copied. Cloud access uses `marin-rigging` bucket settings.

Iris builds use `--backend iris` and `build/ports/service.json`, overridable with
`--iris-service`. Bootstrap an SDK explicitly before target builds:

```sh
uv run --no-project python -m ports.buildomatic.bootstrap \
  --iris-service build/ports/service.json --store build/sdk-client --jobs 128 --python
uv run --no-project python -m ports build python/pillow --backend iris --jobs 128
```

Dependency-ready ports run on separate workers; `--jobs` controls compilation
within a port and `--max-workers` controls port concurrency. Product keys include
build inputs and dependency identities, excluding host tool hashes and scheduling.
Bootstrap preserves completed products and LLVM build state. `--build-key` starts
a fresh request after failure; caches have no retention promise.

Serve a storage prefix locally with `marin-rigging` installed:

```sh
uv run python ports/buildomatic/s3_proxy.py --prefix s3://my-bucket/ports
```

Editable Rigging installs need the Marin working directory. HTTP-encoded GCS
objects return 501.

Existing guest qualification uses `ports.buildomatic.publish`: `check` compiles
native probes, `guest` reruns them, and `release`/`accept` assemble and test a cohort.
That older release format includes the patched WASI uv tool and requires an SDK
bootstrapped with `--resolver`. New wheel/catalog export does not use that tool.

The generic action API is `BuildRequest`/`Action` with local and Iris backends.
Workers mount admitted inputs, run argv, and publish output cache references.
Request journals retain assignments and fence lost workers across retries;
acknowledgement cleans private workspaces. See module APIs for protocol details.
