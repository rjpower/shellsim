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
`--workspace RECIPE=PATH` selects an existing explicit producer workspace whose
inputs the producer must verify. Other adapters currently reject this option.
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
