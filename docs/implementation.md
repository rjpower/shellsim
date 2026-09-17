# Shellsim implementation

Shellsim models a small deterministic Unix machine in one Rust process. Simulated input reaches
only typed state owned by an `Environment`; it never reaches host processes, files, descriptors,
networking, environment variables, or clocks.

## Machine model

```text
Environment
  ProcessTable      logical PIDs, sessions, groups, signals, continuations
  Descriptors       VFS files, bounded pipes, captures, and finite /dev devices
  Vfs               quota-enforced files, directories, links, and metadata
  Timeline          monotonic, wall, and process-CPU clocks plus scheduled events
  VirtualNet        deterministic routes and request records
  Resources         CPU, memory, disk, output, and stop reason
```

The shell parser and expander produce owned syntax consumed by a cooperative executor. Logical
processes retain their complete state across typed waits. A single-threaded FIFO scheduler runs
runnable work and advances virtual time only when all work is blocked. Equal-time events use
stable insertion order.

Pipes, file descriptions, PIDs, queued events, process state, output, and other input-driven
growth are bounded. VFS mutations enforce disk capacity atomically. Resource accounting uses
checked or saturating arithmetic at untrusted boundaries.

## Compatibility boundary

Commands are Rust implementations over a narrow `CommandContext` and explicit `Io`:

```rust
fn run(env: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32
```

Each command is registered as `Real`, `Partial`, or `Unsupported`. Partial and unsupported
invocations remain visible in evaluation telemetry. Registered unavailable commands share one
implementation: they exit 127, write `<command>: not implemented in shellsim`, and appear in
`unsupported`, `unsupported_commands`, and the invocation trace. Unknown options and unsupported
behavior must fail explicitly; they must never invoke a host binary. Native compilation and
arbitrary executable formats remain outside the model because running emitted machine code would
bypass every capability boundary.

Text utilities use a shared option scanner for short clusters, long options, values, and the `--`
operand boundary. The scanner only recognizes command-local option tables. It has no permissive
fallback, so an accepted option always has an implementation in that command.

The standard for a supported command is ordinary usefulness, not exhaustive historical
compatibility. Implement the common behavior as a coherent whole, reject the remaining frontier,
and avoid module-shaped stubs whose successful behavior cannot be explained simply.

## Harness boundary

`serve`, `mcp`, and `replay` all use the same bounded harness operations. The harness owns
persistent environments and exposes execution, stdin and output streaming, VFS operations,
workspace checkpoints and diffs, inspection, cancellation, and deterministic forks. Binary data
is byte preserving. Requests cannot reopen a host path after startup.

`--root` is an explicit trusted harness operation. The importer rejects links and special files,
copies a bounded tree into `/work`, and closes the host boundary before simulated execution. The
copy has no write-back path.

## Adding behavior

1. Read the closest module and its tests.
2. Implement against modeled state and narrow capabilities only.
3. Validate syntax, options, and operands before doing work.
4. Meter loops and reserve input-driven allocation before growth.
5. Return a useful error for the unsupported frontier.
6. Add a unit test for tricky logic and an integration or differential test for observable
   compatibility.

New commands live under `src/commands/`. Machine capabilities belong on `Environment` behind a
small typed interface. Python runtime and module changes follow [python.md](python.md).

## Repository map

```text
src/interp.rs          Environment and persistent shell state
src/process.rs         logical processes and signals
src/scheduler.rs       deterministic runnable and blocked work
src/descriptors.rs     open descriptions and bounded pipes
src/vfs.rs             in-memory filesystem and disk quota
src/resources.rs       limits, accounting, outcomes, and telemetry
src/shell.rs           shell lexer and parser
src/expand.rs          shell expansion
src/exec.rs            cooperative shell execution
src/commands/          modeled command implementations
src/python/            Python source runtime and selected modules
src/harness.rs         persistent agent-session protocol
src/host_ingest.rs     trusted startup-only project import
tests/                 integration, differential, and resource tests
```

## Validation

Run a narrow test while editing. Before handing off a change, run:

```sh
./infra/pre-commit.py --all-files --fix
./infra/pre-commit.py --all-files
./infra/ci/run_tests.py
```

Tests must not depend on host elapsed time, locale, network, filesystem contents, or unordered
collection output.
