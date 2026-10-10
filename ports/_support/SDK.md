# SDK materialization

The default SDK is `wasi-threads-v3`, currently supported on x86-64 Linux. Its
versioned definition is the single source for consumer target, ABI, host compiler
and target platform edges, dependency variants, and pinned producer recipes.

The product graph builds SDK archive tooling and patched host LLVM, then the
threaded libc platform, then threaded CPython when required. The patched host uv
resolver is a separate product. Each node uses its existing producer interface.
Source archives and patches are repository pins, and missing archives are fetched
into the graph source cache. A missing resolver requires its pinned Git source
and the existing online producer; `--offline` rejects it before network access.
An imported verified resolver or an earlier produced resolver can be reused offline. Guest Clang remains a separate requested port.

Host bootstrap configuration contains native executables only. Pass
`--host-seed /path/to/host-seed.json` when products or consumer tools are missing:

```json
{
  "schema_version": 1,
  "host": "linux-x86_64",
  "tools": {
    "cmake": {"path": "/path/to/cmake", "sha256": "<sha256>", "receipt": null},
    "ninja": {"path": "/path/to/ninja", "sha256": "<sha256>", "receipt": null},
    "make": {"path": "/path/to/make", "sha256": "<sha256>", "receipt": null}
  },
  "compiler_tools": {
    "cc": {"path": "/path/to/native-cc", "sha256": "<sha256>"},
    "cxx": {"path": "/path/to/native-cxx", "sha256": "<sha256>"},
    "cmake": {"path": "/path/to/cmake", "sha256": "<sha256>"},
    "ninja": {"path": "/path/to/ninja", "sha256": "<sha256>"}
  },
  "python_helper": {"path": "/path/to/native-python-3.13.7", "sha256": "<sha256>"}
}
```

Replace each placeholder with the executable's SHA256. Paths resolve relative to
this file. Add the adapter's host bindings: `python`, `pkg-config`, `sh`, `rm`,
and Meson for a Meson consumer. Python-backed tools require complete package and
base interpreter proofs; [host tool admission](HOST_TOOLS.md) produces them.
`compiler_tools` may be empty when a verified compiler product exists, and
`python_helper` may be null when a verified Python product exists. These are
native host seeds; no prebuilt target SDK, sysroot, CPython or runtime is required.
The uv producer resolves its pinned native Rust toolchain and records its native
build tools in the resulting producer receipt.

Existing accepted outputs can populate the optional product cache once:

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports._support.import_sdk \
  --legacy-cohort /path/to/old/cohort.json --store /path/to/ports-store
```

This explicit migration verifies original producer receipts, inventories, aliases,
Python headers, runtime relationships and host tool proofs before registering
references. It preserves original receipt hashes and build provenance. It does
not change retained workspaces or claim that newly configured host seeds built
imported binaries. Supplying a helper for a missing downstream product keeps a
verified imported compiler usable. Conflicting existing registry entries fail.
Normal graph builds never load a legacy cohort descriptor.

Run the resulting graph with the ordinary command:

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports native/freetype/graph-recipe.json \
  python/kiwisolver/graph-recipe.json --store /path/to/ports-store \
  --output /path/to/release --check
```

The store records original products under `sdk-products`, producer work under
`sdk-work`, and output manifests under `materialized-sdks`. LLVM retains a Ninja
workspace for compatible source/host seeds; its producer admits patch updates
and seals new immutable products. Other producers receive fresh output attempts
on retry, preserving failed attempts for diagnosis. Product inventories
are checked on every hit. Changed producer policy or dependencies build a new
product; altered cached bytes fail. Each consumer receives compiler/platform
paths and flags plus its own assembled dependency sysroot. Cache identities retain
these actual products, and Python compilation binds headers/configuration.
Resolver and runtime assembly identity changes do not force native recompilation.
The existing format-1 native artifact wire field `toolchain.cohort` retains its
name for immutable released catalogs; new values identify materialized SDK
products, and no setup descriptor is required.

A native command without `--output` needs compiler/tooling/platform only.
`--output` uses the existing combined release format, which includes CPython and
uv. `--check` installs that release through the public API and executes port-local
guest probes before publication. Repeating the command with a new output path
reuses verified results without compiling.

The producer wiring is exercised with bounded inventories at the actual producer
interfaces. A full cold LLVM bootstrap is not established by those fixtures;
accepted LLVM products should be imported rather than rebuilt for orchestration
changes. Simulated programs continue to receive only their virtual capabilities.
