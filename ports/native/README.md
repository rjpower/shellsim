# Shared native target dependencies

The foundation profile links CPython 3.13.7's `zlib` module, Pillow 12.3.0's
PNG codecs and FreeType 2.13.3 against one zlib 1.3.1 target artifact. Pillow
also consumes scalar libjpeg-turbo 2.1.5.1 for JPEG. These target libraries
have no Python distribution metadata.

```sh
uv run --no-project --python 3.13 ports/cpython/build.py \
  --work-dir /tmp/shellsim-native --with-pillow
SHELLSIM_NATIVE_BUNDLE=/tmp/shellsim-native uv run pytest \
  tests/python_package/test_native_dependencies.py
uv run pytest tests/tooling/test_native_artifacts.py
```

`--with-pillow` selects all three libraries automatically. `--with-zlib` builds only the stdlib
consumer. The default profile has no external target libraries. In this static
profile, native package selection happens before the final interpreter link.
The separate [dynamic profile](../dynamic/README.md) loads independent extensions
and shared native dependencies into a fixed interpreter.

## Recipe and artifact contract

`wasi-cpython-v2.json` fixes WASI Preview 1, SDK 34, static linking and standard
Wasm exception instructions. C setjmp/longjmp uses LLVM's SJLJ pass and
`libsetjmp`; C++ uses `-fwasm-exceptions` and the matching libc++/libc++abi/libunwind
sysroot. The final link must also receive `-fwasm-exceptions`. LTO and threads
are outside this profile. NumPy's FFT builtin and its ordinary unique hash path
are enabled; floating-point warning and exception policy remains unsupported.

The SDK 24 v1 profile remains selectable with `--target-profile wasi-cpython-v1`
for bare CPython or pycosat and the existing dynamic C-extension ABI. Its added
empty flag lists do not change that ABI. The migrated native library recipes
require v2; cross-profile artifacts are rejected. Dynamic ABI v1 accepts only
SDK 24 libraries. Dynamic ABI v2 uses SDK 34 with one main-owned C++ runtime
and PIC side modules importing its symbols and exception tag. Typed catches,
rethrows, destructors, RTTI identity and setjmp/longjmp across those modules
have separate guest proofs. See the [toolchain recipe](../toolchain/wasi_sdk/README.md)
for the exact linker contract; generic SDK shared-runtime support is not assumed.

Each library recipe declares its source URL and SHA256, target profile, host
tools, exact target dependencies, selected features, exported headers/archives/
pkg-config files/licenses, approved transitive toolchain flags, and build-script
hashes. Python extension recipes separately declare the target CPython development
configuration and guest distribution metadata. The native catalog uses exact
versions. The host package installer resolves Python version constraints against
a curated set of native wheels; each selected native-library provider has an
exact version and artifact identity.

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

The image manifest records `native_libraries` once and names consumers in
`link_consumers`, with identical zlib artifact identities and explicit archive
inputs. CPython's build profile also records the builder, host driver, native
helper, make, recipe, SDK, and dependency identities. CPython and Pillow refuse
downstream caches whose inputs changed. Use a clean build directory on refusal;
do not erase identity markers while retaining target objects.

## Verification and limits

The guest tests cover compression, PNG/JPEG VFS round trips, recoverable malformed
JPEG errors, scalable font rasterization, invalid fonts, disabled codec libraries
and CPU exhaustion. Wasm fuel meters native instructions. Wasmtime's deferred
reference collector reclaims exception objects. Linear memory and its exception
heap share one aggregate memory bound and one guest reservation; failed growth
rolls back the approved delta. The Wasm GC language proposal remains disabled.
VFS writes retain disk accounting. Target libraries receive no new host capabilities.

The independent SDK probes verify nested setjmp returns, longjmp zero normalization,
C++ typed catch/rethrow/destruction across static archives, one million recoveries
and fuel exhaustion:

```sh
PYTHONPATH=. uv run --no-project python ports/native/exceptions/verify.py \
  --sdk /tmp/shellsim-native/wasi-sdk-34.0-x86_64-linux \
  --work-dir /tmp/shellsim-exceptions
SHELLSIM_EXCEPTION_ARTIFACTS=/tmp/shellsim-exceptions \
  cargo test --test wasm_exceptions -- --include-ignored
```

Compiler guidance: [setjmp/longjmp](https://github.com/WebAssembly/wasi-sdk/blob/wasi-sdk-34/SetjmpLongjmp.md)
and [C++ exceptions](https://github.com/WebAssembly/wasi-sdk/blob/wasi-sdk-34/CppExceptions.md).

The artifacts are integrity records for trusted builds, not an untrusted package
build sandbox. The build host executes reviewed scripts and native helper tools.
Pillow's [imaging profile](../pillow/README.md) does not establish compatibility for
all Pillow APIs or optional libraries. The [shared OpenBLAS port](openblas/shared/README.md)
supplies an independent `libopenblas.so`; the dynamic profile also proves shared
zlib. The [package universe installer](../README.md) resolves Python requirements
and stages the selected native closure without relinking CPython. Arbitrary
native builds and a public artifact service remain outside this prototype.
