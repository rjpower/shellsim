# Deterministic pthread ABI fixtures

This directory pins the static C pthread ABI and its separate diagnostic runner.
The patched `wasm32-wasip1-threads` fixture now executes through shellsim's process
scheduler and virtual descriptors. The published SDK 34 single-thread dynamic
ABI remains distinct. Stock SDK output contains raw atomic waits using Wasmtime's
host blocking/time facilities; shellsim rejects those instructions before
compiling a main or side module, including on cache hits.

## Execution boundary

One virtual process owns a SharedMemory and reserves its fixed maximum size
before allocation. Each virtual thread owns a separate Wasmtime Store, instance,
async continuation, linear stack and TLS. Stores run one at a time in FIFO order;
fuel yields and scheduler host imports suspend a live continuation. No real host
threads run guest code. Ordinary atomic loads/stores/RMW operations remain native
Wasm instructions and run deterministically in this serial polling model.

`wasi.thread-spawn` reserves bounded thread state and queues an independent
`wasi_thread_start(tid, argument)` continuation. It does not execute the callback
inside the spawn call. SDK libc allocates each thread's stack/TLS in the shared
heap and its assembly trampoline installs per-instance stack/TLS globals.

`wasi-libc-scheduler.patch` replaces futex wait/notify instructions with imports
from `shellsim_threads_v1`: `wait32(address:i32, expected:i32, timeout_ns:i64)->i32`
and `notify(address:i32, count:i32)->i32`. Wait returns 0 on notification, 1 for
an unequal value and 2 at a virtual deadline. The value comparison and enqueue
are indivisible scheduler operations; notify wakes a bounded FIFO set. Negative
timeout means no deadline. Clock arithmetic is checked. The patch also replaces
the assembly exit notification and excludes libc's optional busy-wait adapter.
Raw wait32/wait64/notify instructions must be rejected in every executable and
side module before compilation. A future compiler lowering may support third
party intrinsic callers; it is absent from this static ABI.

## Initialization contract

`llvm-serial-memory-init.patch` adds an opt-in `--serial-memory-init` LLD mode,
requiring `--shared-memory`. It preserves the one-time CAS initialization guard
and completed state. It omits raw wait/notify and traps if another initializer is
active. Upstream behavior remains the default. The patch includes a narrow LLD
assembly test for default opcodes, serial opcodes and the incompatible option.

The host must hold a process initialization gate across fuel yields, complete the
main instance first, and serialize subsequent thread/DSO instantiation. It must
publish instance/table slots only after completion and retain failure state.
Blocking constructors, constructors that spawn threads, and reentrant dlopen are
initial unsupported frontiers. Main constructors run once through `_start`;
workers enter `wasi_thread_start`, which does not call `_start`. Each worker must
still receive fresh TLS. The fixture changes initialized process data before
spawn and checks it is preserved, checks a constructor counter remains one,
and checks child TLS starts from its original initializer.

## Remaining foundations

SharedMemory growth bypasses Store resource limiters. Shellsim charges the full
fixed maximum once before allocation, up to 64 MiB. Guest stacks/TLS fit inside
that shared budget. It prepays sixteen reusable host slots, each with a 2 MiB
async stack, 128 KiB of Store overhead and 4,096 table entries at 16 bytes each.
Completed workers drop their Store and free a slot for later workers. IDs remain
monotonic and distinct, bounded by the SDK's positive 29-bit TID range; exhaustion
returns EAGAIN. Compilation scratch, retained code and copied metadata are charged
separately, and reservations are released on process exit or cancellation.

Wasmtime tables/funcrefs belong to their Store. Dynamic modules therefore need a
process registry that allocates stable linear/table offsets and replays the same
module graph into each thread's table using that thread's local functions. Live
load must update all thread tables under the initialization gate. Threaded C++
exception state must use per-thread TLS; the existing non-TLS dynamic runtime is
not a compatible threaded cohort. SDK's p1 thread trampoline is explicitly
non-PIC upstream. These dynamic/TLS issues are outside the first static C probe.

The standalone diagnostic harness executed `two_pthreads.c` successfully with the patched
linker and libc. It checked independent joins, mutex/condition notification,
constructor/data/TLS state and a 5 ms condition timeout driven by virtual time.
It used seven polling turns, three waits and two notifications; virtual time
finished at 5,000,000 ns. The fixture SHA-256 is
`b701c3e74bfce9d0f7065be18d623b2a9c997e237ecf873d0bde9daeae11a7cf`.
Its disassembly contains zero raw wait32, wait64 or notify instructions. The LLD
default, serial-mode and invalid-option checks pass. The unpatched fixture is
rejected by the runner before compilation.

The production process tests also verify this fixture twice with a compilation
cache hit, virtual deadlines, serialized initialization across fuel yields,
blocked-process cancellation, aggregate CPU/memory exhaustion, growth limits and
unequal-value lost-wake prevention. `sequential_pthreads.c` holds fifteen real
workers alive, checks the next create fails with EAGAIN, joins them and completes
forty more create/join cycles using the reusable slots. Worker `proc_exit(0)`
terminates the whole process; an ordinary worker return completes only that thread.
The next milestone is a coherent threaded CPython/runtime cohort and dynamic
TLS/table graph replay, with the existing single-thread dynamic ABI preserved.

## Standalone runner

`runner/` is a bounded harness for this static C ABI, separate from shellsim's
scheduler and kernel. Build it with `cargo build --manifest-path
ports/toolchain/wasi_threads/tests/runner/Cargo.toml`, then pass an explicit patched
fixture path to `shellsim-pthreads-spike`. It rejects raw waits before compiling.
The guest receives captured standard output and a virtual clock, with no host
filesystem, network or clock access. Unimplemented WASI calls trap.

The harness reserves the full 16 MiB shared maximum, uses a 128 MiB modeled
process budget, permits at most 16 threads, and bounds each table to 4,096 entries.
It reserves compiler scratch at 65 times input size plus 4 KiB, then charges the
actual compiled image, twice the source size and 4 KiB. Each thread reserves
512 KiB for its async stack and 128 KiB for Store/table state. Output is capped
at 64 KiB; aggregate polling is capped at 10,000 turns of 10,000 fuel units.
These bounds support the diagnostic; they do not prove production accounting,
cancellation, dynamic loading or CPython thread support.

Use an existing verified standalone patched toolchain for this historical fixture. Run the
patched fixture with `ports/toolchain/wasi_threads/tests/build.py --sdk SDK --toolchain WORK --output OUTPUT`.
The fixture builder verifies the toolchain manifest and rejects raw waits.
Pass `--fixture sequential_pthreads` to build the live-cap and slot-reuse probe.
Fixture inputs are pinned separately from compiler inputs, so adding a probe
does not require rebuilding the unchanged linker and libc. Each probe manifest
records its exact source recipe, compile command, binary hash and instruction audit.
The verified local artifacts and provenance are under
`/tmp/shellsim-threads-spike/{toolchain-manifest.json,proof.json,patched-probe}`.
The original standalone producer build also passed under
`/tmp/shellsim-threads-clean-toolchain`. Its linker, libc archive and final fixture
hashes exactly match the manual build. Its fixture also passed the standalone
pthread execution proof. The clean toolchain manifest records every build command
and the source, compiler, patch, host tool and resource-limit inputs.

The tree cleanup moved the production driver and fixtures without changing the
LLVM/libc sources, patches or C fixtures. Earlier toolchain manifests retain
their original driver identity and are not accepted as new recipe cache entries.
Direct recompilation against those verified binaries is a relocation check, not
a fresh production toolchain build. New recipe builds require fresh provenance.
