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
affected stage. Command substitution still uses a synchronous child adapter, and Python `Popen`
cannot expose a live process. Do not describe those remaining operations as concurrent until
phases 3 through 5 meet their removal criteria.

The readiness handshake needed by phase 3 is present: blocked descriptor operations return a
typed pipe-readable or pipe-writable condition, process code can suspend on that exact condition,
and successful peer I/O wakes matching tasks in stable blocking order. Continuation polling can
therefore yield directly into scheduler state without stringly typed wake keys or host polling.

Shell control flow now runs on a bounded explicit continuation stack. Sequences, boolean lists,
conditions, loops, case selection, functions, positional restoration, and redirection cleanup are
frames polled in fixed work quanta and retained on `ProcessState`. Same-process shell control flow
no longer recurses through the Rust stack. Fork allocations also have independently releasable
memory ownership, which is required once children overlap instead of exiting in stack order.
Ordinary foreground subshells and pipelines now suspend their parent and switch through the
scheduler without a nested Rust executor call. Command substitution, nested-shell adapters, and
Python's synchronous subprocess facade still use the run-to-completion child adapter, so phase 3
remains incomplete.

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
tasks. The Python bytecode VM keeps its instruction pointer and yields when a native modeled
operation blocks.

Completion requires deleting the remaining recursive child run-to-completion adapters. Deep or
adversarial input remains bounded independently of the host stack.

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

## Phase 5: minimal signals and Python `Popen`

Add pending `KILL`, `TERM`, `INT`, `HUP`, `CHLD`, and `PIPE` signals delivered at scheduler
boundaries. Initially only default dispositions are required; unsupported handlers remain explicit.
Add process groups only where foreground pipelines and group termination require them.

Build `Popen` over live process handles and descriptor-backed streams. Implement `poll`, `wait`,
`communicate`, context management, timeout termination, capture, and reaping. `communicate` must
drain both outputs while supplying input so bounded pipes cannot deadlock.

## Phase 6: agent command fidelity

After the process and I/O foundations are complete, improve commands in evidence-driven slices:

1. Resolve VFS executables through exported `PATH`, honor executable metadata and shebangs, and
   distinguish not-found from not-executable failures.
2. Add a bounded `rg` implementation with useful recursive search, glob/type filters, line and
   filename output, fixed strings, regexes, hidden files, binary policy, and familiar exit codes.
3. Expand Git coherently around status, diff, log, show, branch, checkout/switch, add/reset, and
   commits. Repository state stays VFS-only; network remotes remain explicit fixtures or rejected.
4. Add patch application and bounded tar/zip/gzip operations needed by agent workflows.
5. Extend Make only alongside observed syntax and graph semantics; do not emulate native compiler
   artifacts with successful no-ops.

Each command slice records honest trust, validates options, uses semantic integration tests, and
adds a differential test when the reference behavior is deterministic.

## Phase 7: external agent harness

Expose a persistent, host-side protocol that creates an environment from a bounded snapshot,
executes actions with explicit input, reads and writes VFS files, applies patches, returns a
canonical workspace diff, and reports resources, commands, unsupported operations, network
requests, and process state. The Codex or Claude client stays outside the simulation and receives
only these tools. Episode definitions and transcripts must be replayable.

## Validation milestones

At every phase, run the narrow process, shell, time, resource, Python, and pseudo-filesystem tests.
Before a commit, run `./infra/pre-commit.py --all-files --fix`,
`./infra/pre-commit.py --all-files`, and `./infra/ci/run_tests.py`. Do not update fixtures to hide a
semantic regression or retain two competing execution models after a migration phase is complete.
