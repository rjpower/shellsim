# Upstream Python build backends

`python-pep517` executes upstream `build_wheel` hooks for pure source packages and
Wasm extensions. The Kiwi graph uses its upstream setuptools and setuptools_scm
configuration; CPPy's Python helper supplies its own pinned headers. The zss
source package uses the standard setuptools legacy backend for `setup.py`.
That default also applies when `pyproject.toml` has no `build-system` table.
Declared tables must provide a valid `requires` list; absent `build-backend`
selects the setuptools legacy backend.

Select `build_profile: "wasi-threads-v3"` for the threaded target, ABI, host compiler
and platform pins. Declare only additional exact backend wheels as
`build_dependencies`; profile expansion preserves their host roles and explicit
version pins. Dependency variants can use the profile defaults or explicit
`recipe` selections. Package metadata and guest checks remain port-owned. See
[port authoring](../README.md).

Declare exact backend wheels as `build_dependencies`. Include the pinned
`packaging` wheel in that closure: the hook runner uses it to validate requirements.
Backend wheels' runtime dependencies are also part of the host import closure,
so setuptools_scm selects packaging.
An unchanged `pure-wheel` can serve a host build edge and a separately selected
guest root when its complete runtime dependency closure is guest-compatible.
CPPy and setuptools_scm currently cannot be selected as guest packages because
they require the host-only setuptools provider. Their metadata and single
recipe definitions remain unchanged. Build edges alone do not publish or install
their providers in the
guest. CPPy's genuine setuptools runtime requirement remains in its metadata.

Setuptools' universal wheel contains Windows executable launchers. The explicit
`host-wheel` adapter preserves those pinned bytes for host backend imports and
requires `role: "host-tool"`. Such a result cannot be selected as a guest root or
guest dependency. Guest pure-wheel validation continues to reject native bytes.
This adapter does not admit platform-specific host wheels. The current backend
host-tool interface admits a pinned wheel closure and compiler tools only;
additional backend executable or generator dependencies are unsupported.

The backend interpreter comes from the verified
[host-tool descriptor](HOST_TOOLS.md). It runs with `-I -S`, its admitted standard
library, private backend wheel imports, and bounded source-relative
`backend-path` directories. Ambient site-packages, PYTHONPATH and startup hooks
are excluded. Both `build-system.requires` and the optional
`get_requires_for_build_wheel` hook must match the pinned closure, including
active runtime requirements and extras. Missing packages, version mismatches,
and direct URLs fail without downloading dependencies.

The target `_sysconfigdata` file is hash-verified through the admitted CPython
runtime. Its ABI fields and extension suffix are retained; compiler, headers and
linker settings select admitted graph tools and the shared-module link profile.
The result receipt records the overlay, compiler wrappers, effective environment
and backend wheel hashes. Upstream wheel metadata and package layout are checked
against the recipe. Native modules undergo Wasm ABI/provider admission before
normal RECORD sealing. Pure outputs retain universal tags and contain no native
files. Wheel `.data` relocation layouts are rejected. Backends that execute
target binaries during host configuration remain an
unsupported cross-build frontier.

For example, build the current pilot with an already admitted threaded cohort:

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports python/kiwisolver/graph-recipe.json python/packaging python/zss \
  --cohort /path/to/cohort.json --store /path/to/ports-cache \
  --output /path/to/backend-release --check
```

The guest probes solve Kiwi constraints and check translated C++ errors, import
packaging through the public package installer, and calculate zss tree edit
distances. Backend wheels are host build inputs, not guest installations.
