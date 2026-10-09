# Standalone pthread ABI runner

This harness is separate from Shellsim's production kernel and process scheduler.
It proves only the static C thread ABI against a patched SDK cohort. It uses
Wasmtime 49's independent async guest continuations, one process-owned fixed
16 MiB SharedMemory, per-thread Stores/tables/TLS and FIFO polling. No host thread
runs guest code and no host clock drives deadlines. A process initialization gate
is held across fuel yields. Raw atomic waits and notifications are rejected
before compilation; unsupported WASI calls trap when reached.

Limits: 1 MiB input, 16 total threads, 128 MiB aggregate modeled memory, 10,000
scheduler turns with 10,000-instruction fuel quanta, 4,096 table entries per
thread and 64 KiB output. Compiler scratch is precharged at 65 times source
size; retained code uses actual compiled image plus twice source size. Shared
memory's full maximum is reserved before allocation. Each thread reserves its
512 KiB async stack plus 128 KiB for Store/table state; C stacks and TLS live
inside the fixed shared heap. Growth cannot exceed the fixed initial maximum.

Run the pinned Cargo workspace with the final scheduler-safe fixture path. The
runner expects two actual pthread workers, preserved process data and one-time
constructor initialization, independent TLS, joins/mutex/condition behavior,
and exactly 5,000,000 virtual nanoseconds for the condition timeout.

This does not prove dynamic DSO/TLS replay, CPython threading, process integration,
or full cancellation/resource-failure behavior. Those need production work and
additional tests after the static C proof.
