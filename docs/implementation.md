# Shellsim implementation

Shellsim is a deterministic in-process operating environment for evaluating agent-written shell
and Python programs. It models useful behavior with Rust implementations and explicit state. It
does not execute host commands or use the host filesystem as the simulated working environment.

## Architecture

An `Environment` owns all machine and session state:

```text
Environment
  ProcessState     active variables, arrays, functions, cwd, options, jobs, Python REPL
  ProcessTable     bounded PID/PPID records and child lifecycle status
  Vfs              files, directories, symlinks, metadata, disk quota
  PseudoFs         generated read-only /proc and finite /dev views
  Timeline         monotonic/wall/process clocks and scheduled events
  VirtualNet       deterministic route table and responses
  Resources        CPU, memory, disk/output accounting and stop reason
```

Shell source flows through `src/shell.rs`, expansion in `src/expand.rs`, and the executor in
`src/exec.rs`. The executor dispatches commands through `src/commands/mod.rs`. Commands receive a
`CommandContext` plus explicit `Io`; unknown commands resolve through the modeled process `PATH`
to executable VFS scripts and otherwise become recorded compatibility gaps. The native `rg`
implementation performs bounded, metered VFS-only recursive search. Python uses its own source
pipeline described in [python.md](python.md) and shares only modeled environment capabilities.

Logical processes retain forked `ProcessState` values by PID. Shell continuations, background
jobs, sleeps, waits, subshells, and pipeline stages suspend and resume through the deterministic
scheduler. Pipeline bytes flow through bounded descriptor-backed pipes. VFS, clocks, virtual
network, resources, package markers, and telemetry remain machine-wide, so local process mutation
does not leak while filesystem effects remain visible. Exited background records remain until
`wait` reaps them. Background jobs lead modeled process groups, ordinary descendants inherit group
identity, and group signals are delivered at scheduler boundaries. Command substitutions, nested
shells and scripts, Make recipes, `env`, `xargs`,
`timeout`, and the `command` builtin all use retained continuations or scheduled argv children.
Python commands retain compiled code, VM state, output, and top-level bytecode frames in command
continuations and yield after bounded instruction quanta. Ordinary bytecode user-function calls
use explicit return frames as well. Python calls made from within compound native operations still
need instruction-level continuations. Native bytecode calls can now retain normalized arguments
and retry after a typed wake: direct `time.sleep` uses the timer path and unbounded
`Popen.wait()` uses the child path, and unbounded `communicate()` retries on child-tree activity
while retaining partial duplex progress. Direct stream reads and writes wait on exact descriptor
readiness and retain partial write cursors. Timeout-bearing subprocess operations combine those
resource keys with virtual deadline events. The nested scheduler helper is now confined to the
explicitly synchronous `run_python` embedding API.

Reusing an `Environment` preserves the VFS, cwd, variables, functions, arrays, package markers,
virtual time/network state, command history, Python REPL, and cumulative resource usage. `exit`,
`set -e` termination, and CPU/memory/output exhaustion make the session terminal. Disk-full errors
remain recoverable.

## Persistent harness protocol

`shellsim serve` owns one `HarnessSession` and reads bounded newline-delimited JSON requests. The
closed operation set includes one-shot `execute`; retained `start_execute`, `poll_action`,
`write_stdin`, `close_stdin`, `read_action_output`, `signal_process`, and `drop_action`; plus
`read_file`, `write_file`, `remove_path`, `list_paths`, `checkpoint`, `workspace_diff`,
`reset_workspace`, and `inspect`. Each request may carry an arbitrary JSON `id`, which is echoed in
its one-line response. Stream and file bytes are base64; the protocol never performs lossy text
conversion.

A retained action installs a shell continuation and isolated standard descriptors without driving
the machine. Polls execute a bounded number of ordinary scheduler quanta. With `advance_time`
disabled, an all-blocked action returns its typed wait reason without moving the virtual clock;
with it enabled, the scheduler may fire the next modeled event. Streaming stdin is a bounded input
description with explicit append and EOF operations. Output reads use independent delivery cursors
and return only newly available bytes. At most one foreground action is active in a session, up to
64 completed action records may be retained, and an active action survives a complete-state session
fork. The legacy `execute` operation drives this same retained path to completion.

File operations are confined to the simulated `/work` tree. `execute` still sees the full modeled
filesystem and all normal shellsim capabilities, never host state. Requests are capped at 20 MiB,
individual binary transfers and diffs at 6 MiB, and byte decoding is charged before allocation.
Workspace changes are typed, path-sorted added/modified/deleted records containing before/after
file bytes, modes, directories, or symlink targets. A checkpoint clones only a bounded VFS;
`reset_workspace` restores that VFS checkpoint but deliberately does not rewind CPU fuel, virtual
time, process history, shell variables, or terminal resource exhaustion.

Action results include ordered command occurrences and the bounded per-action virtual-network
request delta with method, URL, and
whether a route matched. Inspection returns the retained request log and a dropped-record count;
once the fixed log capacity is reached, later attempts increment that count without growing host
memory.

`serve --root PATH` and `shellsim-python` share `host_ingest.rs`. This trusted startup-only adapter
canonicalizes the selected root, rejects symlinks, special files, and non-UTF-8 names, skips common
dependency/build trees, preserves basic modes, applies file-count and VFS disk limits, and rolls
back the entire import on failure. No request can reopen that host boundary.

`shellsim replay SCENARIO.ndjson` runs the same requests through a fresh session and emits one
typed transcript record per action containing its zero-based sequence number, original request,
and response. A line may instead wrap a request as `{"request": {...}, "expect": {...}}` and
assert exact success, exit status, base64 streams, unsupported, no-op, and partial commands,
workspace change count, or an error substring. Assertion results are embedded in the transcript and
any mismatch makes the replay exit with status 1 after all actions run. Scenario lines remain
subject to the 20 MiB request bound; a replay is capped at 4,096 actions and 64 MiB each of scenario
and emitted transcript data.
`--root` and resource limits have the same meaning as in `serve`, so a checked-in action stream can
be rerun against a bounded host snapshot without changing the protocol. `--transcript PATH` also
persists the emitted NDJSON after a complete replay. Publication is atomic and refuses to replace
an existing path; malformed or incomplete scenarios leave no requested transcript file.

This is sufficient for a host-side agent adapter to replay and assert tool calls without launching
the agent inside shellsim. The library can fork a bounded `HarnessSession`, including scheduler,
descriptor, process, Python, clock, network, resource, and checkpoint state, for deterministic
branching evaluation. Codex/Claude adapters remain harness-side work.

## Simulation boundaries

The VFS and generated pseudo-filesystem facade are the only filesystems visible to simulated code.
VFS mutations are quota-atomic, deletion releases capacity, and the environment supplies virtual
wall timestamps. `/proc` and finite `/dev` nodes are generated from modeled state, consume no disk,
and reject mutation. Commands and Python code must never fall through to `std::fs`, `std::process`,
host environment variables, host networking, or host time.

Time has three domains:

| Domain | Source | Use |
|---|---|---|
| Monotonic | nanoseconds since environment creation | sleeps, deadlines, causal ordering |
| Wall | fixed UTC epoch + monotonic + explicit adjustment | `date`, Python time, VFS timestamps |
| Process CPU | deterministic CPU fuel at 1 microsecond per unit | accounting APIs |

Wall adjustments cannot move monotonic deadlines. CPU work consumes no virtual wall time unless an
operation explicitly models latency. Blocking suspends the current logical process; the scheduler
runs other work and advances to the next event only when its runnable queue is empty. It never
sleeps a host thread. Equal-time events use insertion order.

Native compilation is outside the model. Compiler commands are recorded as `NoOp`: pretending to
compile can keep a setup script moving, but no executable artifact is produced. Running arbitrary
emitted machine code would bypass every modeled capability and resource boundary. The partial
`make` implementation parses a bounded Makefile graph and executes shell-only recipes through the
existing interpreter; it never invokes a host build tool.

## Resource accounting

The resource model is deterministic and approximate:

- CPU is monotonic fuel charged for parsing, dispatch, bytecode, input/output, and algorithms.
- Memory is a modeled working set; command-frame reservations are released when the frame returns.
- Disk is current logical VFS content plus fixed node overhead and is enforced inside `Vfs`.
- Output is a hard cap on materialized stdout and stderr.

Use checked or saturating arithmetic for sizes derived from simulated input. Reserve a conservative
result bound before constructing or mutating large host data structures, and charge loops before
unbounded work. Resource exhaustion returns status 137 with a typed `StopReason`; it must not leave
partially committed VFS or collection state.

Every registered command has coarse base CPU and memory costs. Dynamic implementations add charges
for their actual input and algorithms. Nested dispatch tracks output already charged by children so
the parent does not double-count it.

## Commands and trust

Commands have the uniform signature:

```rust
fn run(env: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32
```

`args` excludes `argv[0]`; stdin is owned input and stdout/stderr are explicit buffers. The central
registry assigns each name a function, base costs, and a trust level:

- `Real`: the supported behavior is intended to be faithful;
- `Partial`: a documented subset such as `sed`, `jq`, or a package facade;
- `NoOp`: pretend-success compatibility with no claimed effect, recorded as unsupported.

Trust is observable telemetry. The environment retains a bounded occurrence log with PID, argv,
trust, completion status, and inclusive CPU/disk deltas; suspended commands remain visible with a
null status. Per-action no-op and partial summaries are derived from that log, so invoking the same
partial command in separate actions remains observable. Do not mark a partial tool `Real`, silently
ignore unsupported options, or invoke a host binary to fill a gap.

## Adding or extending a tool

1. Read the closest command module and its integration tests. Choose an existing behavioral module
   or create a focused file under `src/commands/` with a `//!` overview.
2. Implement the `CmdFn` using only `CommandContext`, `Io`, VFS operations, virtual time/network,
   and nested shellsim dispatch. Never use ambient host capabilities.
3. Register all supported names in the module's `register` function with an honest `Trust` level.
   Add the module to `src/commands/mod.rs` when it is new.
4. Validate options and operands explicitly. Unsupported behavior should return a useful status and
   record the gap rather than producing a plausible but incorrect result.
5. Charge dynamic CPU and reserve memory before reading, expanding, sorting, parsing, or producing
   unbounded data. Let VFS primitives enforce disk capacity atomically.
6. Put small parsing/algorithm invariants beside the implementation. Add integration tests for
   stdout, stderr, status, persistent state, malformed input, exhaustion, and the unsupported edge.
   Use a reference differential when the behavior is deterministic.

When a tool needs a new machine capability, add it to `Environment` as modeled state with a narrow
interface and deterministic tests first. A convenience abstraction is not authorization to expose
the corresponding host facility.

## Repository map

```text
src/interp.rs          Environment and persistent ProcessState
src/process.rs         bounded logical PID and lifecycle records
src/pseudo_fs.rs       generated read-only /proc and finite /dev views
src/resources.rs       limits, accounting, outcomes, command telemetry
src/vfs.rs             quota-enforced in-memory filesystem
src/clock.rs           virtual clocks and bounded event queue
src/net.rs             virtual route-table network
src/harness.rs         persistent typed agent-session boundary
src/host_ingest.rs     trusted transactional host-to-VFS startup import
src/shell.rs           shell lexer/parser and capture API
src/expand.rs          shell word and parameter expansion
src/exec.rs            metered shell executor and control flow
src/commands/          command registry and native implementations
src/python/            Python lexer, parser, compiler, object model, VM, and stdlib
tests/                 cross-module, differential, resource, and acceptance behavior
tests/fixtures/        checked shell/Python inputs and expected outputs
research/              inventories and historical design evidence
```

## Validation

Use a narrow test while iterating. Before committing, run:

```sh
./infra/pre-commit.py --all-files --fix
./infra/pre-commit.py --all-files
./infra/ci/run_tests.py
```

Tests must not depend on host elapsed time, locale, network, filesystem contents, or unordered map
iteration. Compatibility tests should prefer semantic assertions; exact checked output is
appropriate when formatting is the contract. Never weaken a lint, skip a test, or rewrite a
differential fixture merely to make a gate pass.
