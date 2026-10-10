# WASI dynamic loading

## SDK 34 ABI v2

Build a bare SDK 34 CPython bundle, then the fixed dynamic interpreter and
independent proof modules:

```sh
uv run --no-project ports/python/cpython/build.py --work-dir /tmp/shellsim-dynamic-v2/base --build-python /path/to/trusted/python3.13
uv run --no-project -m ports.python.cpython.dynamic --bundle /tmp/shellsim-dynamic-v2/base --output /tmp/shellsim-dynamic-v2
uv run --no-project tests/fixtures/wasm/dynamic/build.py --sdk 34 --bundle /tmp/shellsim-dynamic-v2/base --runtime /tmp/shellsim-dynamic-v2 --output /tmp/shellsim-dynamic-fixtures
SHELLSIM_DYNAMIC_V2_ARTIFACTS=/tmp/shellsim-dynamic-fixtures cargo test --test wasm_dynamic -- --include-ignored
```

The v2 marker is `shellsim-wasi-sdk34-cpython3137-v2`; loader imports use
`shellsim_dylink_v2`. The fixed executable owns SDK libc, libc++, libc++abi,
libunwind and the canonical `__cpp_exception` tag. Standard C library print/scan
support includes long double. The SDK's long-double archive is selected before
default libc through ordinary linker symbol selection, preserving one canonical
definition per runtime symbol. Side modules compile with
`-fPIC -nostdlib -shared --unresolved-symbols=import-dynamic` and import that
runtime. They contain no private C++ runtime. This uses the pinned SDK 34
archives without modifying LLVM or rewriting binary symbols.

The production builder rejects compiled package providers and emits only a
mountable CPython rootfs, its interpreter and runtime provenance. The fixture
builder takes that verified runtime separately and emits test programs,
independent extensions and shared zlib with a proof manifest. Fixture sources
and proof artifacts are absent from the production toolchain recipe and runtime
manifest. The proof manifest records its runtime bundle hash and fixture inputs.

The SDK 34 guest proofs establish normal CPython imports of both C extensions,
then installation and import of a second C/C++ pair while that interpreter is
already running. The interpreter bytes remain unchanged. Two-way
`std::runtime_error` catches and rethrows run each extension's private
destructors. A separate fixture checks a main-owned custom exception's typeinfo
pointer identity, catches in both directions and destructor execution. The
shared zlib consumer roundtrips nonempty and empty bytes through native zlib and
rejects a non-bytes argument with `TypeError`.

V2 resolves declared needed libraries only from `/lib` in the VFS and rejects
dependency cycles. Library identity is its canonical VFS path; distinct package
directories may contain libraries with the same basename. Provider version and
file conflicts are rejected by catalog selection and staging. Symbol lookup searches
the main executable and global libraries, then the deterministic dependency
closure. All side EH tags are imported; C++ runtime symbols belong to the main
executable. Compiler-emitted start initializers run under the process's
metering; external GOT imports must resolve before instantiation. A self GOT
function or data import can resolve to the side module's own export after
instantiation, before relocation functions and constructors. A side module
with a start section cannot use that deferred resolution. SDK 34 mutable
exported globals contain absolute addresses; immutable exported data globals
retain relative offsets. A side
module cannot define a private canonical C++ exception tag or runtime. TLS,
unloading, guest-initiated nested loading and arbitrary native-wheel
compatibility remain outside this proof. SDK 24 libraries cannot enter a v2
process, and SDK 34 libraries cannot enter a v1 process.

V2 permits 32 load attempts and side modules up to 16 MiB. Compiler scratch is
charged at 65 times source length plus 4 KiB before compilation, including cache
hits, and compilation consumes ten CPU units per source byte. Retained costs
include the actual image, twice the source length, 4 KiB and table entries.
Actual linear memory plus EH heap growth is limited to an aggregate 256 MiB
within the environment's memory budget.
The outer Wasmtime async stack is charged at 2 MiB. Each nested host call into
guest code reserves another 2 MiB up to eight simultaneous levels; each
Store retains its high-water charge until completion.

## SDK 24 ABI v1

The spike builds a loader-enabled CPython 3.13.7 interpreter once, then imports two
separately compiled C extensions. The second extension is written into the VFS
from inside the running interpreter before import. Its binary is absent from the
interpreter link inputs. This exercises import-time `dlopen`, Python C API calls,
C method callbacks and independent extension state.

Build against an existing trusted SDK 24 static CPython work directory:

```sh
uv run --no-project tests/fixtures/wasm/dynamic/build.py --sdk 24 --bundle /tmp/shellsim-cpython --output /tmp/shellsim-dynamic
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
`dlopen(NULL, flags)` returns a reserved nonzero main-image handle; `dlsym` on
that handle searches the main executable and globally visible libraries.
Unsupported flag combinations still fail through `dlerror`.

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

## Current v1 frontier

The loader ABI retains SDK 24 and the v1 static profile. Build its base interpreter
with `--target-profile wasi-cpython-v1` or use an existing verified v1 bundle.
The default v1 build driver rejects SDK 34 bundles. Select `--sdk 34` explicitly
for the separate SDK 34 contract above.

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

### Threaded v3 runtime paths

The threaded v3 loader accepts LLVM `dylink.0` runtime-path subsection 5: an
ordered vector of length-prefixed paths, bounded to the module-count limit and
4096 bytes per entry. Dependencies search these directories in the guest VFS,
then `/lib`. `$ORIGIN` and `${ORIGIN}` expand to the importing module's resolved
VFS directory. Each dependency uses its own importer, including nested loads;
resolved VFS paths remain the cache and cycle identities. Host build directories
embedded by Wasm linkers grant no host filesystem access and normally miss in
the VFS. Relative paths and variables other than ORIGIN remain unsupported.
The serial v1/v2 loader retains its existing `/lib` policy.

Runtime-path identity follows the canonical VFS path, rather than the needed
basename: distinct resolved files may load separately. Missing directories or
files continue the ordered search; other VFS errors and non-file candidates
abort loading instead of being hidden by a later fallback.

Threaded v3 resolves weak function imports from existing providers first. An
unresolved function declared weak by LLVM import metadata has a null GOT address
and a signature-matched trapping direct-call binding, matching LLD's undefined
weak stubs. Guarded C++ TLS initialization therefore skips an absent optional
initializer; an unconditional call traps. Missing strong symbols remain errors.
