# A shared syscall boundary for shellsim

## Decision

Keep the virtual kernel in Rust and define its process operations independently of any guest
ABI. A WASI adapter translates guest memory, rights, and errno values into those operations.
Boot a shell as an ordinary logical process by default, selected through the same program loader
as any other executable. Its Rust implementation is a program image, not a privileged process
class. `shell.rs` already owns syntax, `exec.rs` runs commands, and state-changing builtins run
in the shell process. Their source files need not be combined; their authority should be limited
to the shell's process-scoped syscall interface.

The current machine already creates a root process with descriptors and a scheduler entry, and
program continuations belong to logical processes. A child argv loader can retain either a
shell continuation or a native Rust image. The first native images are `pwd`, `true`, `false`,
and `yes`.
They are opaque executable entries in the virtual `/usr/bin`, resolved through the same cwd and
`PATH` search as Wasm files and scripts when launched as external commands. Bare names that are
shell builtins still run in the shell process. The native images receive a
borrowed, process-scoped syscall handle for the current poll quantum; it provides cwd, descriptor
write, and resource accounting, not `Environment` or host handles. Their owned state survives
blocked and partial writes. The shell still boots as the root logical process, and most native
commands still run through the existing dispatcher. It does not yet enforce the stronger boundary:
shell and most native-command code can still mutate `Environment` directly, the WASI adapter
still holds `Interp` while translating imports, and some external native commands execute as
functions in the active process. The target is a loader that starts `/bin/sh`
by default, treats Rust, Wasm, and Python program images as ordinary processes, and runs each
external invocation as a child or `exec` replacement. Generic process state should be separate
from shell-only variables, aliases, functions, job control, and parser state. Builtins such as
`cd` and `export` still act in the shell PID; they do not require kernel privilege.

The host can now create additional persistent shell sessions in one `Environment` and select a
session for each action. Sessions share the VFS and scheduler, but retain separate shell state,
cwd, descriptors, and PID. Idle sessions survive environment snapshots. This is a host control
API, not yet a VFS-loaded `/bin/sh`: the default shell is still a distinguished root process,
and shell execution still has direct `Interp` access.

The typed Rust interface is `System`, borrowed for one execution quantum. It represents the
active PID's view of the virtual kernel, not shell state or a host capability. Native images
receive it directly; the Wasm adapter should translate imports to the same operations. Shell
`cd`/`pwd` and `umask` now use this interface for process cwd and creation mask. The shell keeps
`PWD` and `OLDPWD` variables in its own userland state after a successful `chdir`.
Descriptor reads and writes have typed results; the Wasm adapter uses the same operations for
open virtual files. Its stdio buffering and other imports still need to move to this boundary.

The new `syscalls.rs` begins this boundary for regular files. It accepts typed open options,
allocates descriptors in the active process, and owns close and seek. The WASI adapter now uses
those descriptors for open, read, write, seek, stat, and close; it no longer keeps a second file
handle table. The source-built `guest/wc` fixture proves that a standalone Rust WASI command
can read piped input and virtual files and match selected native `wc` behavior. The fixture is
loaded into the VFS only by tests; standard `wc` remains native.
Native `yes` now runs as a VFS executable and resumes across pipe backpressure through a
process-scoped syscall handle. A typed broken-pipe result determines its exit status without
matching an error message.

For migration, an executable placed at a standard path such as `/usr/bin/wc` takes precedence
over the synthetic native alias. A bare `wc` also resolves that executable through `PATH`; if
no VFS entry exists, the native implementation remains available. This lets a harness opt in to
the compiled command without changing the default base image.

## Contract to extend

| Kernel operation | Guest mapping | Native shell/userland mapping |
|---|---|---|
| Open/close/seek regular file | WASI `path_open`, `fd_close`, `fd_seek` | Redirections and external command I/O |
| Read/write descriptor | WASI `fd_read`, `fd_write` | Process-scoped native handle backed by `Interp::read_fd` / `write_fd` |
| Stat and directory enumeration | WASI `path_filestat_get`, future `fd_readdir` | VFS metadata and shell globbing |
| Spawn/wait/signal | Future versioned shellsim extension | Shell executor and process scheduler |
| Clock/random | WASI clock/random imports | Virtual clock and deterministic stream |

The syscall contract should use typed results, including a wait reason for operations that
would block. The guest ABI layer is responsible only for pointer validation, data copying,
rights checks, and errno translation. It must not perform host I/O or silently return success
for unsupported operations. Resource charges belong at the kernel operation or machine dispatch
point so both native and Wasm work count against the same budget.

## Current limits and next gate

Make VFS program identity authoritative for external native commands, then remove the synthetic
basename fallback. Extend `System` with the remaining typed filesystem and process operations,
and move a reader and a filesystem-walking utility across it. The legacy command registry still
hands most native bodies a `CommandContext` that dereferences to all of `Interp`; remove that
access as commands migrate, rather than renaming it and retaining broad authority.
Finally, give the Rust shell a process-scoped handle for execution, files, and process control;
leave `cd`, variables, and other state-changing builtins in its own PID. Only then should the
default shell be loaded from `/bin/sh` like any other native program image. Native Rust commands
need not become Wasm binaries to use this boundary.

Wasm stdin/stdout/stderr are still buffered at command dispatch. The shell drains a producer's
pipe before starting a Wasm consumer, so a compiled `wc` can count input larger than pipe
capacity and pass its result to another command. It does not prove suspension on a live pipe:
an interactive producer-consumer exchange still cannot run. The next slice should move
stdio onto process descriptors and define a Wasmtime suspension mechanism when a descriptor
returns `IoPoll::Blocked`. A guest continuation must resume at the blocked instruction, not
restart `_start` or treat temporary absence of input as EOF. Before enabling Wasm by default,
resolve how an active Wasmtime continuation can be independently cloned for harness forks, or
reject that fork explicitly. The Rust shell implementation and native `wc` should remain until
those gates pass.

Only after live pipe behavior is sound should shellsim consider a default Wasm `wc`, then a
streaming command and a filesystem walker. WASI Preview 1 alone does not supply Unix
`exec`/`wait` semantics.

## Compiler execution and a future process extension

An earlier WCPL experiment showed that compilation itself requires no special syscall: the shell
resolved the compiler as a VFS executable, and it read source and wrote a Wasm binary through
ordinary virtual file descriptors. That fixture is no longer shipped. The preferred next compiler
is TinyCC. Its pinned package and a wasi-libc subset compile and run C programs inside shellsim;
additional libc behavior still requires corresponding virtual WASI operations.
The current Wasm command path is still buffered, not yet a separately
scheduled guest program image.

If a compiler driver or a Wasm-hosted shell must launch separate tools, add a versioned
`shellsim_process_v1` guest import adapter over typed kernel operations, not a compiler-specific
host callback. The initial contract should accept a validated virtual path, argv, environment,
cwd, and explicit descriptor inheritance/remapping; return a logical PID; and allow the caller to
wait for that PID's exit status. `exec` replacement can be a distinct operation after spawn/wait
works. The native shell would call the same typed kernel operations directly. The adapter must
reject unsupported flags and paths rather than trying the host.

The blocking `wait` operation is gated on a resumable Wasm continuation. When a child is running,
the caller must suspend on a scheduler wait reason while the child makes progress; a blocking host
call or a permanent `EAGAIN` retry loop would deadlock programs that work on Unix. This is why
the extension should follow live Wasm process scheduling, not precede it.
