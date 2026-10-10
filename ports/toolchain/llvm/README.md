# LLVM WASI linker

New threaded port builds use the full [Clang compiler](COMPILER.md). Build that
compiler once and reuse its sealed output through the graph's cohort descriptor.
The linker-only and backend-only profiles below describe the earlier producers
whose outputs remain inputs to existing runtime and sysroot artifacts.

This port builds upstream LLVM/LLD at the exact revision used by wasi-sdk 34.
Two recorded source patches add explicit shared-library initialization and a
serialized shared-memory initializer. Existing linker behavior remains the
upstream default.

```sh
uv run --no-project ports/toolchain/llvm/build.py \
  --archive /path/to/pinned-llvm-project.tar.gz \
  --cc /usr/bin/cc --cxx /usr/bin/c++ --cmake /usr/bin/cmake \
  --ninja /path/to/ninja --work /path/to/fresh-output
```

The source archive, patch input files and production drivers are checked before
building. The build uses four compiler jobs, one linker job, a 12 GiB address
space limit per command and a one-hour limit per command. Use a new output
directory when any input changes. The artifact prefix contains `bin/wasm-ld`
(a relative symlink to `lld`), the upstream LLVM license and `manifest.json`.
The manifest binds the complete recipe to the host compiler/CMake/Ninja hashes,
build commands and produced binary/license hashes. Consumers must verify these
identities and declare the compiler artifact alongside their stock SDK inputs.

## Explicit initialization protocol

`--defer-shared-init` requires `--shared`. Shared-memory libraries additionally
require `--serial-memory-init`. The linker emits no start section and exports
its generated initialization functions when present. It writes a private
`dylink.0` subsection with **uint8** type 128. The payload is the WebAssembly
string `shellsim.deferred-init` followed by varuint32 version 1. This is a
Shellsim initialization contract, not a standardized dynamic-linking feature.
Loaders that do not recognize it must reject the module.

A supporting loader binds external and own GOT references and installs function
pointer slots before calling `__wasm_apply_global_relocs`, then
`__wasm_init_memory` if present, data relocations and constructors. Each threaded
Store initializes its own TLS after the owner has initialized the shared TLS
template. The process owner initializes shared memory, data relocations and
constructors once. Reconstructing a Store never repeats those process phases.
The threaded loader rejects start sections and active data segments and holds
an initialization gate across fuel yields. Blocking or recursive loading from
initializers is unsupported and must be rejected explicitly.

The default SDK linker and older runtime cohorts remain separate. A binary
built with this protocol must be admitted by a matching loader; removing the
start section without the protocol would skip required initialization.

## Threaded compiler profile

`threaded.py` builds a separate `threaded-recipe.json` cohort with `llc` and
`wasm-ld`. It preserves the accepted default linker recipe and artifact. The
threaded recipe adds opt-in WASI dynamic TLS lowering and executable TLS export
classification; it does not change the default static TLS model.

The executable option `--emit-main-tls-info` requires shared memory and forbids
shared-library output. It emits a raw uint8 dylink subsection 129 containing the
vendor string `shellsim.main-tls` and version 1, followed by standard export
flags identifying every exported TLS symbol. The threaded loader requires this
classification and resolves those offsets against each Store's `__tls_base`.
Ordinary data exports retain their normal address convention.

Build the profile with the same host tool and archive arguments as `build.py`,
using `threaded.py` and a fresh work directory. The output manifest records all
four patches, the complete driver/helper identity, host tool hashes, both
compiler binaries and the upstream license. Diagnostic binaries built before
this driver are evidence only and are not admitted as products of this recipe.

The host and guest compiler producers can retain their Ninja trees when a pinned patch is
appended to the existing patch sequence. Admission requires the same source
archive and all non-patch compilation inputs, the exact old patch prefix, an unchanged CMake cache,
and a full comparison of the retained source against the old pinned inputs.
A failed build may receive a verified appended correction after configuration
completed; configuring or unknown phases are rejected. Only a successful
actual build marks the workspace ready.
Only the declared affected files and patch markers move to the new admitted
source. Atomic file replacement and a hash-bound journal make an interrupted
update recoverable; unrelated source edits are rejected. New products record
both source identities and keep the earlier immutable products.

The guest producer also exposes `prepare_guest`: it admits the same inputs,
configures the retained tree and records a Ninja dry-run. Preparation leaves
the workspace unfinished and creates no sealed product. The normal graph
build must complete the actual Ninja actions and seal the new output.

Optional linker-synthesized data symbols belong to the current image. A symbol
exported by a linked shared library cannot supply the main executable's heap or
first-page boundary; explicit definitions in the current object's inputs still
take precedence. The host linker ownership correction also needs to be applied
to the guest linker before guest compilation against shared providers is admitted.

## Canonical executable runtime

Native main links retain one admitted C/C++ runtime. The shared runtime-profile
generator selects the public libc definitions from its GNU archive index and
retains the EH, setjmp and long-double archives before ordinary link inputs.
Shared libraries import process state instead of retaining their own libc.
Executable archive inputs also apply to configure-time executable probes.

Installed guest Clang selects this policy from the versioned
`lib/wasm32-wasip1-threads/shellsim-executable-runtime-v1.json` in its selected
SDK. The driver bounds and validates the profile before replacing `@SYSROOT@`
archive prefixes. The opt-in applies only to wasm32 WASI preview1 shared-memory
links. Compilation and preprocessing do not insert linker inputs. Shared links
omit CRT/default runtime inputs; relocatable and explicitly suppressed-runtime
main links retain the upstream policy.

The canonical main profile requests `--split-runtime-ctors`. This explicitly
opts into the pinned implementation's constructor-priority convention: priorities
through 100 initialize libc and libc++ before side constructors; higher
priorities remain application constructors. The linker exports
`__wasm_call_runtime_ctors`, and ordinary `__wasm_call_ctors` calls this guarded
bootstrap before application constructors. The guard lives in process linear
memory, so worker instantiation cannot reset it. Completed initialization is
idempotent; recursive or concurrent initialization traps under the required
serialized startup contract. The loader binds imports and applies data
relocations before calling the hook, then runs side constructors before the
main command starts. Static mains without this opt-in keep their existing
constructor sequence.

Guest retained builds can update admitted native dependency headers after
verifying the old source, configuration and snapshots. Source patches, target,
compiler, platform/resource headers and build tools stay exact. Atomic header
replacement and an old/new inventory journal permit interrupted updates to
resume; changed files receive fresh mtimes. Ninja recompiles their dependants
before the normal producer seals a new product. Header updates never certify
previous object bytes as outputs of the changed inputs.
