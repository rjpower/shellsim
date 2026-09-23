# A shared syscall boundary for shellsim

## Decision

Keep the virtual kernel in Rust and define its process operations independently of any guest
ABI. A WASI adapter translates guest memory, rights, and errno values into those operations.
The shell remains a privileged, native simulated process that uses the same descriptors,
filesystem, clock, scheduler, and quotas. Its parser and executor need not occupy one Rust file:
`shell.rs` already owns syntax, `exec.rs` runs commands, and state-changing builtins run in the
shell process. Moving that code into a single file would not strengthen the boundary.

The new `syscalls.rs` begins this boundary for regular files. It accepts typed open options,
allocates descriptors in the active process, and owns close and seek. The WASI adapter now uses
those descriptors for open, read, write, seek, stat, and close; it no longer keeps a second file
handle table. The compiled `guest/wc/wc.wasm` fixture proves that a standalone Rust WASI command
can read piped input and virtual files and match selected native `wc` behavior. The fixture is
loaded into the VFS only by tests; standard `wc` remains native.

## Contract to extend

| Kernel operation | Guest mapping | Native shell/userland mapping |
|---|---|---|
| Open/close/seek regular file | WASI `path_open`, `fd_close`, `fd_seek` | Redirections and external command I/O |
| Read/write descriptor | WASI `fd_read`, `fd_write` | `Interp::read_fd` / `write_fd` |
| Stat and directory enumeration | WASI `path_filestat_get`, future `fd_readdir` | VFS metadata and shell globbing |
| Spawn/wait/signal | Future versioned shellsim extension | Shell executor and process scheduler |
| Clock/random | WASI clock/random imports | Virtual clock and deterministic stream |

The syscall contract should use typed results, including a wait reason for operations that
would block. The guest ABI layer is responsible only for pointer validation, data copying,
rights checks, and errno translation. It must not perform host I/O or silently return success
for unsupported operations. Resource charges belong at the kernel operation or machine dispatch
point so both native and Wasm work count against the same budget.

## Current limits and next gate

Wasm stdin/stdout/stderr are still buffered at command dispatch. Thus `wc` demonstrates a real
compiled executable but does not prove suspension on a live pipe. The next slice should move
stdio onto process descriptors and use Wasmi's resumable host-call mechanism when a descriptor
returns `IoPoll::Blocked`. A guest continuation must resume at the blocked instruction, not
restart `_start` or treat temporary absence of input as EOF. Before enabling Wasm by default,
resolve how an active Wasmi continuation can be independently cloned for harness forks, or
reject that fork explicitly. The native shell and `wc` should remain until those gates pass.

Only after live pipe behavior is sound should shellsim consider a default Wasm `wc`, then a
streaming command and a filesystem walker. A compiler guest also needs an explicit process
extension for invoking linker or archive tools; WASI Preview 1 alone does not supply Unix
`exec`/`wait` semantics.
