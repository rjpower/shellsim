# Deterministic time model

## Contract

Time is environment state, never ambient host state. Every effect that observes or waits for time
must name one of three domains:

| Domain | Source | May jump? | Uses |
|---|---|---:|---|
| Monotonic | nanoseconds since environment creation | forward only | causal ordering, sleeps, deadlines, latency |
| Wall | fixed UTC epoch + monotonic + explicit adjustment | yes | `date`, Python `time.time`, VFS timestamps |
| Process CPU | deterministic CPU fuel × 1µs | forward only, independent | Python `process_time`, accounting |

Changing wall time cannot affect a monotonic deadline. CPU work does not implicitly consume wall
time. Pausing for user input does not consume any virtual time.

## Modules and ownership

- `src/clock.rs` owns `Timeline`, clock-domain conversion, bounded event insertion, stable
  `(deadline_ns, sequence)` ordering, ready-event delivery, and task blocking.
- `src/interp.rs` owns the timeline beside the VFS, network, resources, and process state. It also
  carries the deadline interrupt currently unwinding the single synchronous task.
- `src/commands/proc.rs` maps shell `sleep`, `usleep`, `timeout`, and `date` onto timeline
  operations. Durations are parsed exactly at nanosecond resolution.
- `src/vfs.rs` accepts an explicit current wall timestamp from its environment and assigns it to
  creations and content mutations. The VFS cannot read a host clock.
- `src/python/vm.rs` maps `time.time*`, `monotonic*`, `perf_counter*`, `process_time*`, and `sleep`
  to the same domains and deadline path as shell commands.
- `src/resources.rs` defines the stable fuel-to-process-time conversion.

`EventKind::External` is the capability-free ingress contract for future network responses,
watcher notifications, and simulator-driver events. The injector chooses a virtual timestamp and
opaque payload; insertion order breaks equal-time ties. Pending event count and scheduling horizon
are bounded to keep adversarial workloads finite.

## Scheduler progression

The event layer already follows the intended scheduler rule:

1. Run a runnable task.
2. When it blocks, schedule its wake or deadline.
3. If another task is runnable, keep monotonic time fixed and run that task.
4. Only when no task is runnable, jump to the earliest deadline.
5. Move every event at that timestamp to the ready queue in insertion order.
6. Resume affected tasks in stable task-id order.

The current executor has one resumable unit: the foreground command frame. Deadline interrupts
unwind nested shell and Python evaluation correctly. `&` jobs are still evaluated synchronously,
so their sleeps do not overlap. The next executor change is structural rather than temporal:
represent an AST walk as a resumable `TaskFrame`, enqueue background frames, and use the existing
timeline when all frames are blocked. No duration heuristics or host threads are needed.

Runnable bytecode and native commands are zero-duration logical work. Consequently a deadline is
observed when that work reaches a modeled blocking/yield point; CPU fuel, not elapsed time, bounds
a zero-time busy loop. This avoids inventing a machine-speed calibration. A future CPU-duration
model can schedule explicit quantum-completion events without changing the clock domains.

Python's `datetime` and calendar/timezone database are not in the current stdlib slice. They must
fail closed until implemented on top of the wall clock; they must never fall through to host time
or host tzdata. The first calendar policy is UTC/fixed offsets, with version-pinned timezone data
only if task evidence later requires named zones.

## Snapshot and replay

`Timeline: Clone + Eq` includes monotonic time, epoch, wall adjustment, next sequence, pending
events, ready events, limits, and sleep telemetry. A snapshot therefore needs no implicit timer or
host-clock reconstruction. When whole-environment serialization is added, the same fields should
be serialized verbatim and versioned with the environment schema.

External effects must record their chosen virtual timestamp and insertion order in an episode log.
Replay injects those records; it never asks the host when they originally happened.

## Validation

The validation ladder is:

1. Timeline invariants: monotonic non-rewind, stable equal-time ordering, wall/deadline separation,
   deadline-before-wake interruption, bounds, and clone/replay equivalence.
2. Cross-surface integration: shell and Python observe the same epoch and nanosecond advancement;
   timeout prevents effects after its deadline; VFS mtimes equal the wall clock at mutation.
3. CPython/coreutils differential tests for pure conversions and formatting. Absolute `now` values
   are normalized to a supplied epoch; elapsed values and error/status behavior are compared.
4. Existing shell, Python stdlib, pytest/unittest, resource-limit, and TaskTrove suites to catch
   regressions unrelated to time.
5. Future scheduler model-check tests: permute insertion order deliberately, assert stable traces,
   check `sleep A & sleep B; wait` advances by `max(A,B)`, and replay randomized event schedules
   from snapshots.

The tests intentionally avoid elapsed host-time assertions. A virtual year should execute as fast
as a virtual nanosecond except for the deterministic work required to process its events.
