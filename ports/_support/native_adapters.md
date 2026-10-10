# Native build helpers

Port authors call typed helpers in `ports.api` from `build(ctx)`. The canonical
recipe owns static graph dependencies, outputs, exports, patches and guest checks.
CMake/Meson/configure/make options, source preparation and package-specific branches
live in Python. Public helper signatures use explicit named arguments.

```python
from ports.api import BuildContext, meson


def build(ctx: BuildContext):
    return meson(ctx, configure_args=("--buildtype=release",), jobs=2)
```

`ctx.require_native()` returns admitted source, private build/staging paths, target
compiler/archive tools, separate compiler/linker flags and the verified dependency
sysroot. `ctx.sdk` supplies materialized compiler, platform, resources and runtime.
Neither interface admits ambient host paths from guest inputs. These are trusted
host Python builders; they are not an operating system sandbox.

CMake and Meson use admitted generators; configure/make and plain make use admitted
make. Host bindings also include the approved Python, shell, core utilities and
pkg-config. Target searches use the dependency sysroot and selected SDK, excluding
host include/library overrides. Meson disables downloaded wrapped projects.
Upstream installation uses logical `/usr/local` and private `DESTDIR` staging.
Upstream pkg-config files retain their original prefixes.

Helpers return unpublished staging paths and command records. The driver fetches
and checks sources, applies pinned patches, seals exports, verifies ABI/provider
relationships and atomically publishes results. Helper implementation filenames
belong to the shared infrastructure; their content hashes are computed on lookup.
Port-local additional Python helpers are named in static `helpers` declarations.

`NativeArtifact` holds a verified envelope and its payload prefix. `NativeTarget`
binds target, profile, ABI and actual compiler/platform receipt identities.
`merge_dependency_sysroot` verifies the full reachable closure before staging it
under `destination/usr/local`. Conflicting headers, ambiguous SONAMEs, changed
edge identities, cycles, undeclared libraries and escaping paths reject admission.
Unselected results are excluded. Identical nonconflicting file exports can share paths.

Recipe `exports` lists exact payload-relative files; `export_directories` lists
directories by group. Contained installed file links become regular snapshots.
Shared libraries receive the SDK ABI marker and must name their declared direct
providers. Guest tools install through static `install` declarations. Host build
inputs and target-platform products are excluded from guest catalogs.

`build_dependencies`, `platform_dependencies`, `target_dependencies` and
`runtime_dependencies` declare exact provider versions and optional `port:variant`
selections. The SDK adds compiler/platform edges. Ordinary packages retain their
own canonical identity. Runtime edges install guest requirements without adding
link prefixes. Installation requirement ranges remain a separate resolver contract.

Explicit retained Meson trees use
`--workspace python/numpy=/path/to/build/meson-build`. The workspace binds source,
configuration, actual target tools, product receipts and host code. It rejects
unrecorded or incompatible existing trees. Packaging changes can reuse admitted
compiled objects; immutable result identities still bind current builder code.
LLVM has its own admitted persistent Ninja state and product inventories.

The alternative zlib configure and static FreeType Meson demonstrations live in
`tests/fixtures/adapters`. They do not create alternate production package identities.
See [SDK materialization](SDK.md), [Python backends](PYTHON_BACKENDS.md) and
[host-tool admission](HOST_TOOLS.md).
