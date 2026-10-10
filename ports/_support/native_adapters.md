# Native build adapters

The graph runner admits source, patches, cohort, host tools and target dependency
exports before invoking `build_native`. The adapter returns unpublished staging
files and the exact command vectors. Fetching, sealing, cache publication and
catalog assembly remain runner operations.

`NativeBuildContext.target_tools` supplies absolute `cc`, `cxx`, `ar`, `ranlib`
and, for Meson, `strip` entrypoints. Compiler drivers may perform the admitted
cohort's LLVM lowering; the adapter never guesses a compiler from SDK layout.
Compiler and linker flags are separate admitted argument tuples.

The standard host build baseline includes Python, a POSIX shell and core build
utilities (including `rm`), with an explicit receipt supplied by the runner.
CMake needs CMake and Ninja; Meson needs Meson and Ninja; configure/make needs
Make. Every adapter uses a real admitted pkg-config implementation. Host-tool
search directories are constructed from that baseline. Target headers,
libraries and pkg-config files are searched separately and never through host
include or library environment overrides.

Upstream installation uses logical `/usr/local` and `DESTDIR=staging_prefix`.
The runner merges verified dependency exports, with collision checks, under
`dependency_sysroot`. pkg-config receives this sysroot and only its
`usr/local/lib/pkgconfig` and `usr/local/share/pkgconfig` directories. CMake
uses only the dependency sysroot and admitted cohort sysroot for target searches.
Meson disables downloading wrapped projects. Upstream `.pc` files retain their
original prefix; the adapter does not rewrite them.

A recipe build section selects `adapter` (`cmake`, `meson`, `configure-make`),
`configure_args`, `build_targets`, `install_targets`, `jobs`, and
`install_prefix` (`/usr/local`). Arguments and targets are arrays. Configure
recipes provide their upstream-specific cross options explicitly. Meson accepts
its standard `install` target. The runner converts these fields into a typed
`NativeBuildRequest`; it also selects explicit dependency recipe variants.

The compatibility probes build upstream zlib 1.3.1 through CMake and
configure/make, and FreeType 2.13.3 through Meson using the configure-built zlib.
The FreeType archive passes a real guest scalable glyph and malformed-font
probe. These stock-SDK probes establish adapter behavior; they do not certify
threaded dynamic TLS lowering. zlib's upstream CMake script also needs an
explicit WASI library-name portability patch before its generated `-lz`
pkg-config interface can be published.

`NativeBuildContext.shared_library_flags` contains the cohort's side-module
flags. The compiler flag wrapper adds them only when the upstream build requests
its explicit `-shared` mode. Executable compiler checks receive common
`linker_flags` instead.

`native_artifacts.NativeArtifact` holds a prefix and its verified envelope.
`NativeTarget` carries exact target, profile, ABI and toolchain identity.
`merge_dependency_sysroot` validates the direct dependencies and all their
reachable descendants, then atomically writes their exports under
`destination/usr/local`. Extra independent results in the supplied mapping are
not staged. Identical file exports may share a path; differing bytes or a
file/directory collision reject the merge before publication.

`seal_native_install` reads `staging/usr/local`. Recipe `exports` lists exact
payload-relative files; `export_directories` lists payload-relative directories
by group. For example, headers may declare `include` while shared libraries
name `lib/libexample.so` explicitly. Contained file links become regular file
snapshots. Shared library exports receive the exact cohort ABI marker and must
have the declared direct providers in their emitted dependency list. The
artifact envelope retains the original graph recipe and digest separately from
the effective recipe with expanded exact file exports. Source and cache
admission stay with the runner.

Graph recipes distinguish host build programs (`role: host-tool`), target
libraries (`target-library`), installed guest programs (`guest-tool`), and target
platform inputs (`target-platform`). Existing recipes without `role` are target
libraries. `build_dependencies`, `target_dependencies`, `runtime_dependencies`,
and `platform_dependencies` select exact `{port, version, recipe}` providers.
Build edges select host tools; platform edges select target platforms. Target
edges supply link prefixes. Runtime edges supply installed guest requirements
and retain their artifact identities without entering the link prefix.

Native builds declare `toolchain/llvm/host-recipe.json` as a build dependency
and `toolchain/wasi_threads/graph-recipe.json` as a platform dependency. These
small graph recipes pin the existing immutable producer recipes. The runner
verifies their products and provides the selected compiler, sysroot and library
prefixes to adapters. The SDK resource directory remains the admitted source of
compiler-rt builtins, which the host LLVM product does not build.

`--bootstrap` names a JSON descriptor with `schema_version: 1`,
`archive: {path, sha256}`, `work`, and `tools` containing exact `cc`, `cxx`,
`cmake` and `ninja` entries `{path, sha256}`. Paths resolve relative to that
file. The graph calls the LLVM producer with these explicit seed inputs;
verified products in its persistent workspace are reused. Without a bootstrap,
the graph verifies and selects the cohort's existing compiler product.

An in-tree platform source can declare `source.files` as a list of exact
`{path, destination, sha256}` entries. Paths are relative to the ports tree;
destinations are relative to the isolated source directory. `source.sha256`
pins the canonical file list. Only the admitted bytes are staged, and they are
verified before any cached build is reused.

Configure/make recipes may declare `build.configure_environment` for
`ac_cv_*` answers and `CFLAGS`, `CXXFLAGS`, `CPPFLAGS`, `LDFLAGS`, or `LIBS`.
`build.build_args` supplies explicit make arguments to build and install.
These fields cannot replace the admitted compiler, archive tools, shell or PATH.

`llvm-guest` receives the pinned archive directly and owns LLVM source expansion,
patching and verification. Its persistent workspace binds source, patches,
compiler and generator bytes, platform, dependencies and host build tools.
Driver or configuration changes reconfigure the same compatible Ninja tree;
immutable graph results still bind the complete implementation identity.
`--workspace RECIPE=PATH` selects the actual retained Ninja build directory.
The producer state, source and snapshots live in its parent directory, and the
producer verifies their exact inputs before reuse. Without an override, the
runner selects a compatible generated workspace and its `build` directory.
Meson and Python Meson recipes also support retained Ninja directories, as
described in [Retained Meson workspaces](#retained-meson-workspaces). Other
adapters reject this option.
`llvm-guest-sdk` stages admitted target development data without running a
compiler and requires its explicit platform dependency.

A trusted cohort descriptor pins the historical CPython manifest in full.
Admission verifies its runtime files, headers, sysroot build profile and matching
assembled interpreter. Its recipe must match the accepted Python source, patches,
version, ABI and every other policy field. Historical Python driver and JSON
metadata hashes remain provenance in that pinned manifest; their current pins
govern new builds. The declared input paths must match exactly, and compiled
facade sources, headers and patches must retain their current accepted hashes.

The `wasi-sysroot` platform node invokes the pinned libc producer when
`--platform-bootstrap` supplies a schema-1 descriptor with `sdk_archive` and
`libc_archive` objects (`path`, `sha256`) and a `work` path. Paths are relative to
the descriptor. The node consumes its explicit host LLVM build dependency and
cohort-admitted CMake/Ninja tools. A complete existing workspace product is
reused only after exact recipe, compiler, tool and output-byte verification.
Without bootstrap archives, the explicit cohort platform receipt is reused.
Guest LLVM workspace compatibility tracks compiler and generator bytes and
consumed headers; changed linker tools or archives update verified snapshots
at stable paths and trigger relinking. Result identities still bind all files.

Each native graph node seals a resolved toolchain receipt with the actual
compiler and platform manifest hashes. Its cohort identity derives from the
input cohort and those two receipts. Graph acceptance retains that per-node
context and compiles probes with the same selected compiler and sysroot; it
also records the resolved receipts in `graph.json`. Bootstrap-selected products
therefore cannot retain the original cohort toolchain label.

Port checks may declare `test_limits` with positive integer `cpu`, `memory` and
`disk` guest budgets. Omitted fields retain the public environment defaults.
The harness caps declarations at one trillion CPU units, 16 GiB memory and 2 GiB
disk, and records the effective explicit budget in the acceptance receipt.
These budgets allow compiler and SDK checks to include installation and execution
costs. Temporary release materialization lives under the proof directory and is
removed after the checks.


## Python Meson projects

The `python-meson` adapter uses an admitted host Python and Cython for generators,
and admitted CPython headers and `pyconfig.h` for target compilation. Its
pkg-config wrapper answers target Python queries with those headers; other
providers resolve only through the dependency sysroot. Optional
`build.host_header_packages` entries bind header-only generator packages to a
complete host-tool receipt, including the package files beyond its executable.

`build.cross_properties` supplies upstream cross facts.
`build.dependency_properties` resolves a property from an admitted native port
and export-relative directory. `build.install_tags` defaults to runtime,
python-runtime and devel. Meson's install plan selects the build targets and
supplies wheel paths, extension names and directory exclusions. The adapter
removes host SOABI suffixes, preserves qualified package paths, copies upstream
PKG-INFO as METADATA, and binds each Wasm extension to the target ABI and its
actual shared-library imports. No maintained extension inventory is required.

`build.development_exports` maps installed wheel files or directories to the
native payload beneath `/usr/local`. These exports use the same sealing and
dependency closure as native libraries. NumPy exports its generated C headers,
libnpymath archive and upstream pkg-config files from the same build as its
wheel. A mixed wheel/development port installs both selections for acceptance.

The scientific host descriptor admits a Meson tree patched with
`meson-wasi-archive-groups.patch`. Meson otherwise inserts GNU archive groups for
WASI Clang. wasm-ld rescans archive members and rejects these GNU flags. The
receipt binds NumPy's vendored Meson source, the patch input and output hashes,
and every resulting tool file. [Scientific host-tool setup](HOST_TOOLS.md)
produces these receipts with a private read-only Python closure.


## Retained Meson workspaces

`--workspace RECIPE=PATH` also accepts Meson and Python Meson recipes. PATH is
an actual Ninja directory named `meson-build`; its parent holds compiler
wrappers, while the enclosing directory holds admitted source and dependencies.
For example, use `--workspace python/numpy/graph-recipe.json=target/numpy-work/build/meson-build`.

The first build requires a fresh directory. Existing trees without a workspace
receipt are rejected. Each resume verifies exact source bytes and executable
modes, configuration, target product receipts, host tool code, compiler flags
and dependency exports. Receipts also bind both generated compiler wrappers,
the target Python pkg-config launcher when used, and the effective build
environment. Effective setup arguments, cross and native machine files,
properties, installation prefix and fixed setup flags are also bound, including
options appended by Python Meson. Only the installation destination (`DESTDIR`)
is excluded; paths in compiler or configure bindings remain exact. A changed
compilation input rejects that workspace without deleting it. Recipes with
mutable hooks cannot use this mode.

Jobs, install tags, licenses and development-export selections remain packaging
inputs rather than compilation-workspace inputs. The runner still keys and
verifies every immutable result with the full recipe, implementation and cohort.
A packaging change reuses the verified Meson configuration, runs Ninja's selected
install targets and writes a fresh install and wheel directory. Unchanged
compiler wrappers and machine files retain their timestamps so Ninja does not
reconfigure or rebuild generated headers.
Each retained build copies its verified workspace receipt into the sealed
result provenance. Older receipt schemas are rejected and require a fresh
workspace. Result-cache hits, workspace decisions and phase
timings are printed to stderr.

Host f2py is an optional explicit binding in both Meson cross and native files.
Its complete admitted NumPy package supports SciPy's generation steps; target
NumPy headers and f2py C sources come from the native dependency exports.

Shared links keep `-nostdlib` so libc, pthread and interpreter state come from
the process runtime. The runner supplies the target compiler-rt builtin archive
from the verified SDK as an explicit trailing link input. This resolves numeric
compiler helpers, including quad-precision conversions, without relying on
which helpers a particular interpreter link happened to export. The archive
path and hash remain result inputs; retained workspace admission also binds
these trailing inputs, so a changed link policy cannot reuse stale modules.

The one-module Python adapter accepts `build.output: stdlib` for CPython's own
extensions. It compiles the pinned CPython source against admitted headers,
seals `lib-dynload/<module>.so` and licenses as a native artifact, and emits no
wheel metadata. Additional internal include directories must stay below the
admitted CPython `Include` tree. An internal graph runtime edge can select such
an artifact without changing an upstream package's `Requires-Dist`. Publication
verifies its native closure, then assembles a separate runtime with the module
and providers. It preserves the base interpreter and records exact artifact
identities; module files are never published as generic `/lib` providers.

Shared links treat linker warnings as errors, including incompatible function
signatures. Python Meson packaging compares direct imported function types
against the actual exports of declared shared providers before sealing.
`build.required_shared_libraries` names admitted provider basenames that at
least one installed extension must declare in its Wasm dependency metadata.
This establishes the provider link separately from behavioral guest checks.

The pinned Meson patch also handles WASI link checks with explicit shared
provider inputs. It links those checks as PIC side modules and rejects
unresolved symbols. Executable checks keep their normal policy. OpenBLAS
symbol-existence checks retain volatile function addresses without calling
unknown prototypes; callable ABI checks still reject signature mismatches.

Compiler wrappers read LLVM GNU response files for option classification,
including nested `@file` arguments. They pass the original argv unchanged to
the compiler. Compile-only modes take precedence over shared-link selection.
Expansion is bounded to 16 nested files, 256 files, 4 MiB and 65,536 arguments;
unreadable, cyclic or excessive inputs fail before compiler invocation.

Native build commands bound Git discovery at the admitted source and build
parents. Extracted tarballs therefore use upstream version fallbacks instead
of the enclosing Shellsim commit. A pinned checkout inside the source tree
remains discoverable. Inherited Git directory/work-tree overrides are excluded
by the existing target environment allowlist. Retained generated VCS headers
may change once when this boundary corrects a previously embedded parent hash.

`build.executable_cohort_link_inputs` declares at most 32 static archives by
canonical path relative to the resolved platform sysroot, for example
`lib/wasm32-wasip1-threads/libsetjmp.a`. The runner rejects escaping paths,
symlinks, missing files, duplicate entries, archives larger than 128 MiB, and
bytes absent from the admitted platform inventory before consulting the result
cache. Exact paths and hashes remain result inputs. The common compiler wrapper
adds these archives after the original executable link arguments, preserving
static archive order. Compilation and shared links do not consume them. Retained
workspace admission binds the resulting compiler wrapper text.

Native acceptance selects executable dynamic linking when its declared exact
link inputs include a shared provider. Static-only probes keep the ordinary
executable link policy. This selection is derived from admitted files rather
than arbitrary test linker switches; the resulting command records the exact
inputs and still runs as a native guest executable.

Executable archives also apply to executable links made by upstream configure
probes. Their feature checks see the same executable archive inputs as the final
commands. Shared links receive their independently declared shared inputs.
