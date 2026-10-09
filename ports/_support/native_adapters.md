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
