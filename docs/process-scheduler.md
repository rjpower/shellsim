# Cooperative process and agent-shell roadmap

Shellsim will model a small deterministic Unix machine, not host processes or a general kernel.
This document is the implementation contract for moving from synchronous child-state swaps to
cooperatively scheduled logical processes, then improving the command surface most useful to
coding agents. A phase is complete only when its old special-case path is removed and its success,
failure, exhaustion, and unsupported boundaries are tested.

## Implementation status

Phase 1 is complete. `scheduler.rs` provides bounded FIFO runnable, running, blocked, exited,
wake, and reap transitions. `descriptors.rs` provides bounded shared open descriptions,
per-process descriptor maps, VFS file/input/capture/null endpoints, and bounded pipes with
backpressure, shared cursors, endpoint lifetime, broken-pipe behavior, and EOF. Every process owns
an FD table, children inherit shared descriptions, `/proc/PID/fd` is generated from that table,
and shell commands route input and output through descriptors. Redirections are transactional and
ordered, including descriptor duplication, close, append, invalid input paths, and the standard
`/dev` descriptor aliases. The public finite-buffer execution API is now only a harness adapter
that installs and captures descriptors.

Phase 2 is complete. Complete `ProcessState` values now live in a machine-owned PID map. Creating
a child retains the parent in that map, and scheduler dispatch changes the active PID rather than
moving parent state through recursive executor frames. Exit removes the finished execution
context while the process table independently retains zombie status until reaping.

Phase 3 is in progress. Shell control flow and ordinary subshells run as stored continuations.
Background commands are inserted as runnable tasks, return before execution, retain their fork
allocation while alive, and make detached terminal output available exactly once after they later
run. The scheduler switches the active PID between retained process contexts in one-instruction
quanta. Shell `sleep` and `usleep` register resumable command entry points: they block their task
on a virtual-timeline event, allow other tasks to run, and advance time only when the runnable
queue is empty. Pipeline stages are created together and exchange bytes through bounded pipe
descriptors; command input and output are resumable frames so backpressure suspends only the
affected stage. Nested `sh`/`bash` script execution from an ordinary shell continuation now starts
a scheduler-owned child and resumes through a typed child-wait state. Executable shell scripts
resolved through the VFS use that same child path, including buffered standard input. Python
`Popen` now launches the same stored argv continuations and drives them at a nested cooperative
scheduling boundary, with live PIDs and bounded descriptor pipes. Command substitutions in
commands, redirects, here-documents, `for`, `case`, arithmetic commands, and every phase of
C-style `for` loops launch scheduler-owned captured children. `source` and `eval` inject parsed
frames into the current continuation, preserving their same-process semantics without nesting the
executor on the Rust stack. The old recursive captured-child adapter has been removed. Python
commands now retain compiled code, VM state, and buffered output in typed command continuations;
bytecode yields in bounded quanta and CPU-bound Python processes interleave in FIFO order. Ordinary
user Python calls use explicit callee/return frames and retain exception unwinding across them.
Callbacks made from within compound native operations and blocking native methods still execute
through nested Rust calls, so suspension is not yet available at every bytecode instruction.
Direct `time.sleep` calls are the first native suspension path: they register a typed timer wait,
retain the active Python call frame, and resume after the scheduler wakes that process.

The readiness handshake needed by phase 3 is present: blocked descriptor operations return a
typed pipe-readable or pipe-writable condition, process code can suspend on that exact condition,
and successful peer I/O wakes matching tasks in stable blocking order. Continuation polling can
therefore yield directly into scheduler state without stringly typed wake keys or host polling.

Shell control flow now runs on a bounded explicit continuation stack. Sequences, boolean lists,
conditions, loops, case selection, functions, positional restoration, and redirection cleanup are
frames polled in fixed work quanta and retained on `ProcessState`. Same-process shell control flow
no longer recurses through the Rust stack. Fork allocations also have independently releasable
memory ownership, which is required once children overlap instead of exiting in stack order.
Ordinary foreground subshells, pipelines, and shell-command invocations now suspend their parent
and switch through the scheduler without a nested Rust executor call. Python subprocess operations
use live handles and the scheduler rather than the removed synchronous runner, but the calling VM
remains on the Rust stack while the scheduler runs child quanta. Command substitution is now part
of the retained shell continuation and preserves its capture and child identity across timer and
descriptor waits. Native command adapters can now launch a typed sequence of scheduler-owned argv
children; `env` uses it with an isolated launch environment and `xargs` uses it for ordered
invocations. `timeout` schedules typed signal delivery to the child's retained process tree and
reaps that tree after expiry. Make evaluates its bounded dependency graph into an ordered recipe
plan, then runs each recipe as a scheduler-owned shell child; a failed recipe stops the sequence.
VFS Python shebangs now cross the same scheduled child boundary, and the `command` builtin injects
an argv continuation into its current process instead of recursively dispatching. The Python
bytecode VM remains the main phase 3 execution gap.

## Non-negotiable invariants

- Simulated input never reaches a host process, descriptor, filesystem, network, environment, or
  clock capability.
- Scheduling is single-threaded and deterministic. Equal-priority work runs in stable insertion
  order; no behavior depends on host timing.
- Process count, descriptor count, pipe capacity, continuation state, captured output, and queued
  events are bounded and charged before growth.
- A PID identifies one retained process record from creation through reaping. Parentage, cwd,
  environment, state, and exit status remain observable through the generated `/proc` view.
- File-descriptor duplication shares an open description and therefore its cursor and status
  flags. Fork copies descriptor entries, not underlying resources.
- Pipe EOF occurs only after the final writer closes. Empty reads with live writers and full writes
  with live readers suspend instead of manufacturing data or completing eagerly.
- Unsupported executable formats, syscalls, signals, shell syntax, and tool options fail visibly.
  They never fall through to an ambient implementation.

## Target machine model

```text
Environment
  Vfs, Timeline, VirtualNet, Resources
  ProcessTable<ProcessId, Process>
  OpenDescriptions<DescriptionId, OpenDescription>
  Pipes<PipeId, Pipe>
  Scheduler { runnable, blocked, current }

Process
  pid, ppid, process_group
  argv, cwd, environment, shell state
  FdTable<Fd, FdEntry>
  Continuation
  Runnable | Blocked(reason) | Exited(status)

OpenDescription
  VfsFile { node, offset, access, append }
  PipeReader(PipeId) | PipeWriter(PipeId)
  Null | Capture(OutputId)
```

The scheduler polls a runnable process until it exits or returns a typed blocking reason. If no
process is runnable, it advances the virtual timeline to the next event and wakes affected tasks.
It does not spin, sleep a host thread, or use nondeterministic work stealing.

## Phase 1: descriptor foundation (complete)

Introduce a bounded descriptor arena and per-process `FdTable`. Seed descriptors 0, 1, and 2 with
explicit input/output captures. Model VFS open descriptions, null, and captures before pipes.

Route existing redirections through ordered descriptor operations. Opening all redirections must
be transactional: failure leaves neither partial files nor a partially mutated descriptor table.
Support close and duplication for ordinary shell forms, including order-sensitive `2>&1 >file`
and `>file 2>&1`. Replace `/dev/stdin`, `/dev/stdout`, and `/dev/stderr` write special cases with
descriptor resolution, and generate `/proc/PID/fd` from the table.

Completion removes executor-local `RedirPlan`, eager descriptor aliases, and direct standard-stream
path handling. Tests cover shared offsets, append, duplication order, close, invalid descriptors,
fork inheritance, descriptor exhaustion, output limits, and pseudo-filesystem visibility.

## Phase 2: stored process contexts (complete)

Move complete child `ProcessState` values into machine-owned process entries. Replace
`start_child`/`finish_child` swapping with explicit creation, activation, exit, and reaping APIs.
The root session remains a retained process. Machine capabilities stay outside process state.

Define typed process states and wait reasons. Fork-state accounting remains bounded, and failure is
atomic. Existing subshell, command substitution, pipeline, nested-shell, and Python subprocess
tests must pass without a compatibility adapter that recursively swaps the active process.

## Phase 3: resumable execution

Represent shell execution as explicit frames rather than the Rust call stack. A poll performs a
bounded amount of work and returns `Ready(status)` or `Pending(BlockReason)`. Preserve sequence,
condition, loop, function, redirection, `errexit`, and cleanup semantics in frame data.

Keep ordinary native commands synchronous. Convert only commands that may block into resumable
tasks. Python code, operand stacks, exception regions, and top-level instruction state are now
owned by a retained command continuation. Ordinary bytecode calls now push explicit return frames.
The next executor change must split compound native operations around their Python callbacks, after
which retryable subprocess and descriptor operations can return the same typed
ready/block/switch result as shell frames. Direct Python timer sleeps already use this result path.

The shell-side recursive child adapters are removed from normal execution. Python operand,
scope, exception, context-manager, and method state is now separated from the VM's temporary
environment borrows, and executing code objects own their code, instruction pointer, and handler
stack in explicit bytecode frames. Ordinary user-function dispatch pushes and returns through that
owned frame stack without Rust recursion. Compound operations such as constructors, sorting keys,
and protocol callbacks retain synchronous inner calls for now. Phase completion requires reifying
those callbacks and deleting the nested scheduler bridge used by blocking Python methods. Deep or
adversarial input must remain bounded independently of the host stack.

## Phase 4: deterministic scheduler and pipes

The FIFO runnable queue, typed wake keys, bounded pipe objects, reader/writer reference counts,
backpressure, and asynchronously runnable background shell continuations are present. The next
timer waits, event-loop clock advancement, scheduler-backed `wait`, concurrent pipeline stage
construction, and descriptor backpressure are present. Native commands still consume a bounded
complete input value and produce a bounded complete output value internally, but their descriptor
read and write frames suspend incrementally, so pipe capacity is not bypassed.

Required compatibility cases include overlapping sleeps, file races with deterministic ordering,
`yes | head`, early reader close, multi-stage pipelines, blocked writers, pipeline status and
`pipefail`, job status transitions, nested deadlines, and resource exhaustion without deadlock.

## Phase 5: minimal signals and Python `Popen` (baseline complete)

Pending `KILL`, `TERM`, `INT`, `HUP`, `CHLD`, and `PIPE` signals now live on process state and are
delivered at scheduler boundaries. Terminating signals wake blocked tasks, use conventional
`128 + signal` statuses, and cancel abandoned timer events; `CHLD` is coalesced with its default
ignored disposition. The shell `kill` builtin supports PID/job targets, supported signal names and
numbers, existence probes, and signal listing. Custom handlers and process groups remain explicit
future work.

`Popen` is built over live process handles and descriptor-backed streams. `poll`, `wait`,
`communicate`, signal termination, context management, timeouts, capture, and reaping use real
logical child state. `stdin`, `stdout`, and `stderr` expose small binary/text file-like facades.
`communicate` drains both outputs while supplying input, and retains its write cursor across a
timeout so retry cannot duplicate input. Unsupported host setup and session options fail before
launch. Full process groups, custom signal handlers, and arbitrary Python-bytecode suspension are
outside this baseline.

## Phase 6: agent command fidelity

After the process and I/O foundations are complete, improve commands in evidence-driven slices:

1. Resolve VFS executables through process `PATH`, honor executable metadata and supported
   shebangs, and distinguish not-found from not-executable failures. This foundation is complete.
2. Add a bounded `rg` implementation with useful recursive search, glob/type filters, line and
   filename output, fixed strings, regexes, hidden files, binary policy, and familiar exit codes.
   This baseline is complete; unsupported options fail explicitly.
3. The coherent Git baseline now covers human and porcelain status, staged and working-tree diff,
   name-only diff, bounded history, show, refs, branch creation/listing/deletion, checkout/switch,
   restore, reset, add, and commits. Repository state stays VFS-only; merges, remotes, and network
   operations remain explicitly unsupported.
4. Atomic, bounded VFS-only patch application is present for unified diffs and the common
   `*** Begin Patch` agent format. Deterministic VFS-only `gzip`, `gunzip`, and `zcat` support
   bounded compression, decompression, files, and streams. Tar creation, listing, extraction, and
   gzip composition support regular files and directories with atomic traversal-safe extraction.
   ZIP creation, listing, and extraction support stored and deflated regular files plus directories,
   with CRC validation and the same bounded, atomic traversal protections.
5. Extend Make only alongside observed syntax and graph semantics; do not emulate native compiler
   artifacts with successful no-ops.

Each command slice records honest trust, validates options, uses semantic integration tests, and
adds a differential test when the reference behavior is deterministic.

## Phase 7: external agent harness (in progress)

The `HarnessSession` library and `shellsim serve` NDJSON frontend now retain an environment across
explicit actions, base64 VFS file operations, checkpoint/reset, typed canonical workspace diffs,
and resource, command, unsupported-operation, trust, and process reports. Requests and transfers
are bounded; the line-oriented exchange is directly replayable. The Codex or Claude client stays
outside the simulation and receives only these tools.

Bounded transactional host-directory ingestion is shared by `serve --root` and
`shellsim-python`. Per-action and retained virtual-network request reporting is also present.
Bounded NDJSON scenario replay emits paired request/response transcript records. Scenario
assertions, session cloning, and concrete model-client adapters remain before this phase is
complete.

## Validation milestones

At every phase, run the narrow process, shell, time, resource, Python, and pseudo-filesystem tests.
Before a commit, run `./infra/pre-commit.py --all-files --fix`,
`./infra/pre-commit.py --all-files`, and `./infra/ci/run_tests.py`. Do not update fixtures to hide a
semantic regression or retain two competing execution models after a migration phase is complete.
