# Shared native target dependencies

The foundation profile links CPython 3.13.7's `zlib` module and Pillow 12.3.0's
PNG codecs against one zlib 1.3.1 target artifact. zlib is a native-library
provider and has no Python distribution metadata.

```sh
uv run --no-project --python 3.13 ports/cpython/build.py \
  --work-dir /tmp/shellsim-native --with-pillow
SHELLSIM_NATIVE_BUNDLE=/tmp/shellsim-native uv run pytest \
  tests/python_package/test_native_dependencies.py
uv run pytest tests/tooling/test_native_dependencies.py
```

`--with-pillow` selects zlib automatically. `--with-zlib` builds only the stdlib
consumer. The default profile has no external target libraries. Native package
selection happens before the final static interpreter link; the public pure-wheel
installer cannot add a new native library to an assembled interpreter.

## Recipe and artifact contract

`wasi-cpython-v1.json` fixes WASI Preview 1, SDK 24, static linking, and the absence
of threads and C++ exception support. The SDK supplies libc and compiler runtime
libraries. The profile does not expand NumPy's existing FFT or floating-point
exception support.

Each library recipe declares its source URL and SHA256, target profile, host
tools, exact target dependencies, selected features, exported headers/archives/
pkg-config files/licenses, approved transitive toolchain flags, and build-script
hashes. Python extension recipes separately declare the target CPython development
configuration and guest distribution metadata. The native catalog uses exact
versions; this small implementation has no version-range solver.

An artifact lives in `native-artifacts/<input-sha256>`. Its `artifact.json` records
the recipe, source archive and consumed source-tree hashes, verified build-script
hashes, SDK identity, compiler/sysroot content identity, dependency artifact hashes,
and every exported file's hash. Its artifact SHA256 covers that whole manifest.
Builders verify all existing exports before cache reuse and reject extra files,
symlinks, missing files, or different inputs. A changed recipe, source tree,
toolchain, or dependency produces a different artifact identity.

`dependencies.py` creates a consumer prefix from the declared closure only. It
rejects missing providers, conflicting versions or target profiles, conflicting
export paths, cycles, and changed transitive artifact identities before staging.
It supplies archives before their dependencies in the final link sequence.
Only the pinned SDK can supply approved toolchain flags such as `-lm`.

Target builds discard ambient `CPATH`, `C_INCLUDE_PATH`, `LIBRARY_PATH`, compiler
flags, and pkg-config search overrides. CPython receives explicit zlib flags and
disables pkg-config probing. zlib and Pillow invoke the SDK compiler directly,
with explicit target headers and archive paths. They execute no build-system
download or host-library discovery. The declared source and SDK archives are
fetched and checked before compilation; source pins are not signatures.

The image manifest records `native_libraries` once and names both consumers in
`link_consumers`, with identical zlib artifact identities and explicit archive
inputs. CPython's build profile also records the builder, host driver, native
helper, make, recipe, SDK, and dependency identities. CPython and Pillow refuse
downstream caches whose inputs changed. Use a clean build directory on refusal;
do not erase identity markers while retaining target objects.

## Verification and limits

The guest tests compress and decompress data, write a PNG to `/tmp` in Shellsim's
VFS, reopen it, and compare size, mode, and pixels. They reject invalid compressed
and image data and verify disabled optional codec libraries. An unbounded zlib
compression loop must stop at the guest's cumulative CPU limit. Wasm fuel meters
the library's native instructions; linear memory and VFS writes use the existing
memory and disk accounting. Target libraries receive no new host capabilities.

The artifacts are integrity records for trusted builds, not an untrusted package
build sandbox. The build host executes reviewed scripts and native helper tools.
Pillow's [PNG profile](../pillow/README.md) does not establish compatibility for
all Pillow APIs or optional libraries. Dynamic loading and a general package
resolver remain separate work.
