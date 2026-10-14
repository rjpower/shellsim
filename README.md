# shellsim

Shellsim is a BusyBox for containers: one small, deterministic process that provides a useful
Unix-shaped environment without starting a VM, container runtime, or host subprocess. It is built
for experimentation and testing with reinforcement-learning rollouts and agentic environments,
where fast startup, reproducibility, isolation, and explicit resource limits matter more than
cycle-accurate emulation.

Shell programs, common command-line tools, logical processes, and Python run in-process against an
in-memory filesystem. Simulated code cannot access the host filesystem, processes, network,
environment, or clock. A trusted harness may copy a selected project into the virtual filesystem
before execution; changes never write back to the host.

## Compatibility

Shellsim aims for broad compatibility inside clear boundaries. A supported facility should handle
almost all ordinary uses, even when obscure flags or legacy behavior remain out of scope. A module
or command with no coherent useful subset is omitted instead of being exposed as a misleading
stub. Unsupported syntax, options, executable formats, and capabilities fail visibly and are
included in structured results.

The current environment includes:

- a Bash-like shell with pipelines, redirections, functions, common expansions, control flow,
  background jobs, signals, and job control;
- common filesystem, text, archive, Git, Make, process, and system commands;
- deterministic virtual time, network fixtures, `/proc`, `/dev`, processes, descriptors, and
  bounded pipes;
- upstream CPython and third-party packages supplied by the WASI ports catalog.

Python runs as a fuel-metered Wasm executable. The catalog supplies extensions built for its
CPython/WASI ABI; package preparation happens on the host before execution. Shell-only machines
have no Python installation. See [Python in shellsim](docs/python.md).

`git` provides a porcelain subset backed by the virtual filesystem: staging, commits, history,
diffs, branches, tags, merges, stashes, and search, with output that matches real Git where an
agent is likely to parse it. Networked subcommands and content conflicts are refused rather than
approximated. See [Git in shellsim](docs/git.md) for the supported surface and its boundaries.

Common utilities accept clustered short options, long options, option values, and `--` through a
shared parser. Each utility declares its supported options. For example, `grep` supports ordinary
basic and extended regular expressions, recursive search, fixed strings, word and whole-line
matching, pattern files, include and exclude filters, context, counts, line numbers, match limits,
and common output controls. `sed` covers addresses and ranges plus the common substitution,
selection, text, transliteration, and early-exit commands. `awk` parses a typed language subset
with record rules, control flow, fields, associative arrays, arithmetic, regular expressions, and
the usual scalar functions. Unsupported syntax and options exit nonzero with a direct diagnostic;
text processing is UTF-8-only unless a command documents a byte-oriented mode.

## Install and run

Install the Python package and console command:

```sh
python -m pip install shellsim
shellsim -c 'printf "b\na\n" | sort'
shellsim --root ./project -c 'ls /work'
```

For C/C++ builds, select the Clang port with `ctx.Environment(tools=["clang"], memory=8 * 1024**3)`.
See [guest Clang](docs/guest-clang-performance.md) for catalog setup.

`--root` copies the selected host tree into a disposable `/work` snapshot. With no `-c` and a
terminal attached, `shellsim` starts a persistent interactive session.

To build the Rust binaries from source:

```sh
cargo build --release
./target/release/shellsim -c 'echo hello'
./target/release/shellsim eval --cpu 100k --memory 8m -c 'make test'
```

Limits accept `k`, `m`, and `g` binary suffixes. `eval` emits a structured result containing the
exit status, stdout and stderr, resource use, command trace, and unsupported behavior.
`unsupported_commands` lists recognized or attempted commands that crossed the capability
boundary; invocation records include trust, status, and the unsupported reason.

The Python API exposes fresh and persistent environments:

```python
import shellsim

environment = shellsim.Context().Environment(cpu=100_000)
environment.write_file("/work/answer.txt", "42\n")
result = environment.run("cat /work/answer.txt")
assert result.returncode == 0
assert result.stdout == b"42\n"
```

`environment.run()` reuses the default shell. To keep independent shell state in one machine,
create additional sessions; they share the virtual filesystem and resource limits:

```python
first = environment.create_shell()
second = environment.create_shell()
first.run("cd /work; export LABEL=first")
assert first.run("printf '%s' \"$LABEL\"").stdout == b"first"
assert second.run("printf '%s' \"${LABEL-unset}\"").stdout == b"unset"
```

Static HTTP fixtures use the same isolated environment. They do not enable sockets, DNS, TLS, or
host-network access:

```python
environment = shellsim.Context().Environment(http={
    "https://api.test/items/*": shellsim.HttpResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body='{"items": []}',
    ),
})
result = environment.run("curl -s https://api.test/items/1")

assert result.network_requests[0].matched
```

Run Python and compiled packages using an exported [ports catalog](ports/buildomatic/README.md):

```python
context = shellsim.Context()
environment = context.Environment(pypi_packages=["scipy"])
result = environment.run_python("from scipy.special import factorial; print(factorial(5))")
result.check_returncode()
```

This mounts upstream CPython 3.13.7 WASI and downloads only the selected wheels,
native dependencies, and interpreter. Context reads `DEFAULT_PYPI_CATALOG` on
the first port request; pass `pypi_catalog` to override it. Cache, offline and
precompiled trust policy belong to the Context. Identical selections reuse
an immutable installation with a private filesystem for each guest. Create a new
Context to see catalog updates. No patched uv is needed;
`pypi_packages=()` selects CPython without extra packages; `install_pypi` can add catalog
packages later. Each Python invocation starts a fresh interpreter on the persistent VFS.
See [CPython WASI ports](ports/README.md)
for capability limits and the explicit `CPythonRuntime` API.

## Clock modes

A machine runs on virtual time by default: when every process waits on a timer, the scheduler
jumps the clock to the next deadline, so `sleep 3600` costs no host time and runs are
reproducible. An embedder can boot a machine with
`Environment::with_limits_and_clock(limits, ClockMode::RealTime)`, or with the CLI's
`--real-time` option, for interactive demos. Its
clock then follows physical time from boot: shell, Python, and Wasm sleeps take real time, and
clock reads report elapsed time. Polling never blocks in either mode. An idle real-time machine
reports that it is blocked, and `Environment::host_wait_until` says how long to wait. Simulated
programs cannot choose the mode, and a snapshot of a real-time machine is not reproducible.

## Experimental Wasm executables

An executable VFS file beginning with the WebAssembly magic bytes runs through a fuel-metered
Wasmtime engine. The WASI Preview 1 adapter provides standard streams, arguments,
exported environment variables, the virtual clock, deterministic random bytes, and basic regular
file access relative to the virtual working directory. It does not expose host files, processes,
network, environment, or time. Invalid modules and imports outside the WASI namespace fail with a
diagnostic. Unavailable WASI calls trap if reached; they never report fake success.

Shellsim vendors the Wasmtime 49.0.0 crate with one runtime patch:
`CallHook::FuelResume` runs after a fuel yield, before guest execution resumes.
Threaded CPython uses this boundary to install newly loaded shared modules and
callback bindings into each thread's Wasmtime Store. Upstream's host-call hooks
do not cover a thread suspended by fuel while computing. See
[the Wasmtime overlay rationale and maintenance contract](docs/wasmtime-overlay.md)
for the patch, provenance, reproduction, and packaging limits.

A Wasm executable runs as its own scheduled process on that process's virtual descriptors. A
read or write that would block suspends the guest until the pipe or terminal is ready, so a guest
can answer a prompt, sit in the middle of a pipeline, or be stopped by `kill` or `timeout`. A
write to a pipe with no reader ends the guest with status 141. Fuel yields keep a compute-bound
guest from starving other processes, and consumed fuel counts against the machine's CPU budget
while the guest runs. A harness fork taken while a guest is running gets a copy of that process
that fails with status 126 and a diagnostic, since a live Wasmtime stack cannot be cloned.
Clock subscriptions to `poll_oneoff`, which back libc `sleep` and `nanosleep`, block the guest on
virtual time like the native `sleep` command.

This is not full WASI process support. Descriptor subscriptions to `poll_oneoff` fail with
`ENOTSUP`; sockets and guest process creation remain unsupported. No C compiler is installed by default. The engine accepts Wasm
exception instructions. A pinned external TinyCC package and permissively licensed wasi-libc
sysroot can be installed into the VFS by a caller; integration tests compile and run C programs
through that virtual toolchain. This is a tested subset of libc, not full POSIX support. A prebuilt
`wasm32-wasip1` command that uses only the listed imports can be written to the VFS with
executable permissions and invoked by path or through `PATH`.

Wasm compilation uses a host-private persistent cache with an 8 GiB soft disk limit and a
bounded process-local module/admission cache. Host cache state does not change simulated
resource charges. See [guest Clang loading measurements and cache settings](docs/guest-clang-performance.md).

The [virtual display](docs/virtual-display.md) lets a Wasm guest present RGBA frames and poll
bounded key events. A host-driven session can resume the guest between frames without granting it
ambient display, input, or network access.

## Agent harness

`shellsim serve --root ./project` runs a persistent newline-delimited JSON session. It supports
bounded execution, streaming actions, VFS operations, checkpoints, workspace diffs, process and
resource inspection, and deterministic session forks. `shellsim mcp` exposes the same environment
as a stdio MCP server. `shellsim replay scenario.ndjson` reruns checked action transcripts.
`shellsim corpus manifest.json` runs frozen shell and Python workloads under a versioned stock
environment and emits a classified JSON report. See [compatibility testing](docs/compatibility-testing.md).

`shellsim serve --host-tools` grants a virtual guest-to-host tool route at construction. The
trusted runner receives typed calls in `poll_action` and completes them with `respond_tool_call`;
ordinary sessions have no such route. See [`.shl` package design](docs/shl-packages.md).

Python hosts can instead use `shellsim.Container(tools={"name": handler})` to register handlers
at construction and call `container.run(...)` without a CLI subprocess or manual polling.

```sh
printf '%s\n' \
  '{"id":1,"op":"execute","source":"printf hello > result"}' \
  '{"id":2,"op":"workspace_diff"}' \
  | shellsim serve --root ./project
```

## Resource model

CPU is deterministic fuel, memory is modeled working set, disk is current virtual-filesystem
usage, and output bounds materialized stdout and stderr. The defaults are 100,000,000 CPU units,
64 MiB memory, 64 MiB disk, and 4 MiB output. Costs use simple, stable approximations intended to
be correct within an order of magnitude. They bound runaway work and make runs comparable; they
do not model allocator layouts or instruction timing exactly. Exhaustion is observable and never
falls back to an ambient host implementation.

Every CLI command that boots a machine accepts the same options: `--cpu`, `--memory`, `--disk`,
and `--output` (counts accept `k`, `m`, or `g` suffixes), and `--real-time`. They may precede the
command, as in `shellsim --cpu 1m -c 'make'`. Commands without script arguments (`shell`, `eval`,
`serve`, `mcp`, `replay`) also accept them after the command name.

For internals and contribution workflow, see [implementation](docs/implementation.md),
[Python](docs/python.md), and [CONTRIBUTING.md](CONTRIBUTING.md).
