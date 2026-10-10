# CPython virtual process port

This port adds ordinary `os.pipe`, `os.waitpid`, `os.posix_spawn`, and
`subprocess.Popen` entry points to the fixed CPython 3.13.7 WASI interpreter.
They call the versioned `shellsim_posix_v1` virtual kernel. A child is another
simulated process; no guest call launches a host process.

`process_abi.h` defines the wire layout and its 64 file-action slots. Private
`process_limits.h` bounds spawn and exec to 4,096 arguments, 256 environment
entries and 131,072 combined string bytes, including terminators. This policy
header is compiled into the process implementation rather than toolchain clients. File actions run in order on the child's descriptor table before exec:
close, dup2, closefrom, open, and chdir. Open and chdir paths are copied when
the action is added and freed when the action list is destroyed. The kernel must
repeat bounds and pointer checks before creating a child. Reserved WASI
preopens 3 and 4 are adapter identities, separate from user descriptors.
The kernel accepts UTF-8 arguments, environment entries and paths; paths are
limited to 4,096 bytes. Shebang resolution is bounded to eight levels.
`waitpid` supports a positive child PID or `-1`, with `WNOHANG` as its only
option. Process group waits and stopped-child waits are unsupported.
`kill(-1, signal)` is unsupported; `kill(0, signal)` and a negative process
group ID address virtual groups.

`cpython-3.13.7-process.patch` exposes the port through upstream CPython's
`posixmodule.c`. It lets `subprocess.py` use POSIX spawn for a WASI child,
including `cwd`, default signal restoration, and `close_fds`. Options that
require fork or unsupported process identity/session attributes raise before
launch. A canonical `signal` wrapper syncs supported default and ignored
dispositions with the kernel. Custom handlers are an explicit unsupported
frontier; the CPython patch leaves SIGINT at its default on WASI. Default
`restore_signals=True` restores SIGPIPE in the child and requires waitpid to
report signal death separately from a numeric exit code.
For SIGBUS, SIGILL, SIGFPE, SIGABRT and SIGSEGV, the fixed interpreter retains
the pinned WASI libc's synchronous `signal()`/`raise()` callbacks. CPython's
`faulthandler` uses that path, so raised fatal signals print a traceback.
External virtual `kill` does not deliver those callbacks, and WebAssembly
traps are not converted into C signals. Fatal SDK `raise()` ends as a Wasm trap,
which shellsim reports as a failed program rather than a POSIX wait signal.
Virtual `getpid` and `getppid` replace SDK 34's constant PID stub in the fixed
interpreter. Process identity, shebang resolution and executable format checks
come from the virtual kernel; a shebang replaces the child image under the same
virtual PID.

SDK 34 defines `O_CLOEXEC` as zero. This fixed interpreter uses a reserved
`0x00080000` flag in its process libc and CPython posix object. `pipe2`, `dup3`,
`open`, and `openat` set virtual descriptor inheritance accordingly. Ordinary
`open` and `openat` use the versioned `descriptor_open` import to install the
descriptor and its flags atomically, including when guest threads can yield.
The profile accepts read/write/search access, directories, final-component
`O_NOFOLLOW`, creation, exclusivity, truncation, append and nonblocking mode.
Other flag capabilities, including synchronous I/O, return `ENOTSUP`.
Ordinary
side modules built against the unmodified SDK still see its zero constant;
they must use `fcntl(F_SETFD, FD_CLOEXEC)` when they need this behavior. A side
module that statically links SDK's emulated `getpid` also retains its stub;
ports needing virtual process identity must link to the fixed interpreter's
canonical symbol.

`exec.c` supplies canonical `execve`, `execv`, `execvp`, and `wait` for native
tools that link this additional facade. Exec copies bounded UTF-8 argv and
environment vectors, resolves the supplied PATH, validates the target image,
then replaces the same virtual PID. Failed validation leaves the old image,
environment and descriptors intact. Success stops all old guest threads and
never returns. Buffered display sessions reject exec before replacement.
The existing CPython overlay does not yet expose `os.exec*` through its platform
configuration. `tempfile.c` supplies bounded `mkstemp` using virtual random
bytes and exclusive creation; it never uses host files or a nonexclusive
temporary-name fallback.

The build overlay preserves the input bundle's native extension ABI. From the
repository root, with the pinned source bundle already built:

```sh
uv run --no-project --python 3.13 python -m ports.toolchain.wasi_process.build \
  /path/to/fixed-bundle /path/to/process-bundle
```

The recipe pins the CPython source files, patch, SDK and port inputs. The
output manifest records the input bundle, linked objects, interpreter and
patched standard library hashes. `tests/python_package/test_cpython_process.py`
is an opt-in guest acceptance test for pipes, spawn, signals, timeouts and
unsupported frontiers. The process bundle requires matching
`shellsim_posix_v1` kernel imports and descriptor readiness.
