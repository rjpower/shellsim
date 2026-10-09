# SDK 34 dynamic runtime contract

The recipe defines `shellsim-wasi-sdk34-cpython3137-v2` for a fixed CPython
3.13.7 executable and independently linked wasm32 WASI shared modules. It pins
the SDK archive, build utilities, fixtures and runtime notices. The builder
records actual tool and runtime archive hashes; it rejects changed reviewed
inputs before compiling.

The main executable links the SDK's libc, libc++, libc++abi and libunwind
archives once, with the SDK's long-double print/scan implementations selected
before default libc. Complete C symbols are retained through linker undefined
symbol requests; runtime definitions remain unique. Side modules use PIC code and `-nostdlib` with LLVM's
`--unresolved-symbols=import-dynamic`, importing the main exception tag and
runtime. Linking the SDK's non-PIC C++ runtime archives into a shared module is
unsupported. No LLVM patch or binary symbol rewriting is required for this
contract.

The main also links the pinned `posix.c` adapter before libc. It defines the
complete SDK duplication symbol group and `fcntl`, so ordinary archive selection
omits the SDK's unsupported implementations. Shared libraries call the same
canonical exports. The adapter requires `shellsim_posix_v1.descriptor_control`;
the bundle records `runtime_capabilities: ["shellsim_posix_v1"]`. An older
Shellsim backend rejects this import explicitly. The package dynamic ABI remains
v2 because its memory, table, exception and C runtime ownership contract is unchanged.

The same namespace provides `umask`, `cwd_get` and `cwd_set`. Canonical libc
`getcwd` and `chdir` share a bounded 4096-byte cwd with the SDK's path resolver.
A constructor initializes it from the virtual launch cwd; successful `chdir`
updates the kernel process cwd and the WASI adapter together. Root preopen 4
remains `/`. Relative paths and later child processes therefore use the same
virtual directory. No environment variable overrides this state.

The builder enables upstream CPython's `os.umask` by compiling its unchanged
`posixmodule.c` with `HAVE_UMASK` against the declared canonical libc prototype.
Only this compiled object is replaced in the normal main link. The base build,
`pyconfig.h` and source tree remain unchanged; the manifest records source,
original object, replacement object and compiler-command provenance. The
virtual process mask applies to file and directory creation.

Permission reporting remains a WASI frontier: Preview1 filestat has no mode
field, so CPython's SDK-backed `os.stat` synthesizes permission bits instead of
returning the VFS's stored mode. The kernel still enforces the creation mask;
the guest stat result is not an authoritative permission check.

Duplication shares virtual open descriptions and file cursors. CLOEXEC is a
per-process descriptor flag: `dup` clears it, `F_DUPFD_CLOEXEC` sets it, fork
copies it, and successful exec closes flagged aliases. `dup2(fd, fd)` preserves
the entry. Append status belongs to the shared description and survives aliases
and fork. Null-device reads return EOF and writes discard bytes, with the
requested read/write access enforced. Descriptor 3 and 4 remain reserved WASI
preopens; this profile rejects replacing them. Unsupported fcntl commands and
nonzero dup3 flags fail explicitly. The adapter owns no host handles.

WASI file resizing uses the VFS disk budget, meters growth before allocation,
and preserves cursor and unlinked-file identity. Temporary growth reservations
are released on failure and success. Redirected standard descriptors expose
their actual file metadata and seek state.

SDK `VERSION` identifies LLVM commit
`895aa2c896ada719451be2e3673c83da8ddf1141` and wasi-libc commit
`2e6fb9d8ee0cdf9e431fbcabe8af3115de000a13`. The checked-in notices come from
those commits. Allocator notices are extracted from the recorded source line
range; both source and notice hashes are retained. The rootfs carries these
notices under `/TOOLCHAIN-LICENSES`.

Build and guest proof commands are in [the dynamic port](../../dynamic/README.md).
