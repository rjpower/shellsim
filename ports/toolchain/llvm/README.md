# LLVM WASI linker

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
