# Experimental WASI dynamic loading

The spike builds a loader-enabled CPython 3.13.7 interpreter once, then imports two
separately compiled C extensions. The second extension is written into the VFS
from inside the running interpreter before import. Its binary is absent from the
interpreter link inputs. This exercises import-time `dlopen`, Python C API calls,
C method callbacks and independent extension state.

Build against an existing trusted SDK 24 static CPython work directory:

```sh
uv run --no-project ports/dynamic/build.py --bundle /tmp/shellsim-cpython
SHELLSIM_DYNAMIC_ARTIFACTS=/tmp/shellsim-dynamic cargo test --test wasm_dynamic -- --include-ignored
```

The script leaves the source bundle untouched. It writes a new interpreter,
a tiny C command and library, two extensions and a manifest under
`/tmp/shellsim-dynamic`; `--output` changes that destination. The manifest records
the interpreter and extension hashes. The standard static CPython bundle and
public wheel installer retain their current profiles.

The tiny command checks shared writable data, a callback into the executable,
constructor initialization, data relocations, repeated handles and missing
symbols. The integration tests also check an incompatible ABI, missing Python
API imports, local versus global symbol scope, library count, extra memory,
unsupported dependency metadata, compilation costs on cache hits and an infinite
constructor consuming its CPU budget. Tests using built C artifacts require the
opt-in environment variable; synthetic loader tests run in the normal suite.

## ABI and resource boundary

The executable and every library carry a `shellsim.abi` custom section containing
`shellsim-wasi-sdk24-cpython3137-v1`. This exact marker
identifies the experimental compiler/runtime contract; it is not a signature or
an assertion that arbitrary marked binaries are compatible. The executable
imports `open`, `symbol` and `error` from `shellsim_dylink_v1`, exports its memory,
function table, mutable C stack pointer, `malloc` and required C API symbols.
The C bridge supplies `dlopen`, `dlsym`, `dlerror` and a process-lifetime `dlclose`.

Libraries use LLVM's wasm32 `dylink.0` layout and import the executable's memory,
stack and table. The loader allocates aligned data through guest `malloc`, grows
the shared table, resolves `env`, `GOT.mem` and `GOT.func`, applies data relocations
and calls constructors. Data symbols are relocated by each library's memory
base; function symbols refer to the shared table. Direct imports search the main
executable, then libraries opened globally. GOT imports also find the library's
own exports. `dlsym(handle, name)` searches that library; a zero handle searches
the main executable and global libraries. Reopening a local library with
`RTLD_GLOBAL` promotes its symbols.

Each process allows at most 32 library load attempts and reserves 32 MiB for
loader metadata, source bytes, compiled artifacts and additional table entries.
The conservative retained cost is 65 times each source length plus 4 KiB, with
16 bytes per added table entry. A library file is capped at 4 MiB and its metadata
at 1 MiB. Compilation costs ten CPU units per source byte even on cache hits;
data clearing, symbol table scans and all guest initialization are metered.
Guest data allocations remain inside the command's existing bounded memory
reservation. The loader reservation is released with the process, including
cancellation. Libraries receive the same virtual WASI boundary as the main
module; loading never reads a host library or grants host execution capabilities.

## Current frontier

The loader ABI retains SDK 24 and the v1 static profile. Build its base interpreter
with `--target-profile wasi-cpython-v1` or use an existing verified v1 bundle.
The build driver rejects the SDK 34 exception profile before relinking. SDK 34's
C++ exception runtime does not support shared libraries, so the v2 static
cross-archive exception proof does not extend this dynamic ABI.

The profile retains libraries and data for the process lifetime; `dlclose` does
not unload them. A failed load disables further loads in that process because
failed initialization can leave guest data or table entries behind. Missing
symbols and ABI errors are reported through `dlerror`; exhausted virtual budgets
terminate execution. Nested loading, automatic `DT_NEEDED` dependencies, runtime
search paths, nonempty symbol-flag metadata, TLS, weak symbols, module start
sections and separately defined memories/tables are rejected. `RTLD_NOLOAD`,
`RTLD_NEXT` and bare-name library search are absent. No C++ exception ABI or
arbitrary native-wheel compatibility has been established.

This initial loader proves independent C extensions. Shared zlib dependencies,
NumPy as a side module and a host package installation ABI need separate build
and guest evidence.

The format follows the upstream
[LLVM dynamic linking convention](https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md).
