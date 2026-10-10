# Buildomatic core

This package runs trusted host build actions. It has no Iris dependency and does
not expose host capabilities to shellsim guests. The ports bridge owns recipe
translation and publishing; backends own deployment and remote storage.

`BuildRequest(idempotency_key, actions, max_workers=1)` admits at most 512 nodes
and 32 workers. `Action(id, argv, dependencies=(), inputs=(), env=(),
timeout_seconds=3600, max_attempts=2)` runs argv directly. IDs and mount names are
single safe path components; a bridge can hash canonical port references. One
action can run bounded Ninja or make parallelism within a port.

Each `InputMount(name, TreeBundle(digest))` expands under `inputs/<name>` in a
private workspace. Successful dependencies mount under their action IDs.
`BUILD_INPUT_<name>` and `BUILD_OUTPUT_DIR` provide absolute paths. The designated
output directory is `output`; intermediates belong elsewhere in the workspace.
Worker argv starting with `python` or `python3` resolves to `sys.executable` on
the worker. The environment starts with PATH, HOME and TMPDIR plus explicit
action entries. No shell is used unless the trusted argv explicitly selects one.

```python
from pathlib import Path
from ports.buildomatic import Action, BuildRequest, Coordinator, LocalStore, WorkerExecutor

store = LocalStore(Path("build-store"))
worker = WorkerExecutor(store, Path("build-worker"))
coordinator = Coordinator(store, "request-123", {"local": worker})
coordinator.submit(BuildRequest("request-123", (Action("example", ("python", "-c", "print('hello')")),)))
# Call tick repeatedly to reconcile workers and dispatch ready actions.
result = coordinator.tick()
```

Tree manifests reference immutable SHA256 chunks of at most 4 MiB. Manifest and
blob reads have a 32 MiB bound, enforced by stores before allocating beyond that
bound. Files, directories, executable permission bits and internal relative
symlinks round trip. Extraction rejects traversal, linked parents, link cycles,
special files, inconsistent chunk sizes and corrupted digests. Capture requires
a quiescent tree. Failed extraction removes the new destination.

`ResourceLimits(memory_bytes=8*1024**3, cpu_seconds=3600,
output_bytes=2*1024**3, max_files=100000, log_bytes=8*1024**2)` configures bounds.
Raise `output_bytes` explicitly for SDKs larger than 2 GiB. It bounds expanded
tree transport, the total mounted inputs and workspace files excluding inputs.
Transport does not load entire product or SDK files into memory. Memory and CPU
rlimits apply per subprocess; disk and log bounds are monitored during execution
and checked at completion. These are trusted workloads, not hostile programs or
a container isolation boundary. The local subprocess executor requires Linux.

`Store` supplies `put_blob(data)->digest`, `get_blob(digest)->bytes`,
`read_journal(key)->VersionedBytes|None` and
`write_journal(key, data, expected_version)->version`. Blob writes verify
content and are immutable; reads reject corruption and oversized data. Journal
CAS uses opaque backend revisions, with `None` meaning create-if-absent. Even
identical writes must receive a new revision. Conflicts raise
`ConditionalWriteError`. A remote store can use native conditional objects.

One journal key owns one accepted request. Identical submissions return its
identity; conflicting content or keys raise `IdempotencyConflict`. The journal
stores the accepted request separately from output cache blobs. Opening its
coordinator claims a new generation, fencing old coordinators with
`CoordinatorFenced`. A service lease should serialize coordinator ownership.
`read_build_result(store, journal_key)` reads a bounded typed journal snapshot
without writing or claiming a generation. It returns `None` only for an absent
journal and rejects malformed data. `request_id(request)` returns the public
canonical request identity without accepting or dispatching the request.
Attempt IDs and assigned workers are durable before dispatch. Worker RPC errors
retain assignments; they never imply loss. Explicit `UNKNOWN` permits a
bounded retry, after cancellation tombstones fence delayed dispatch. A worker
endpoint removed from the mapping remains uncertain until restored.

`Coordinator` accepts 0..32 available workers. An empty pool leaves undispatched
nodes pending while still applying cancellation and confirmed instance retirement.
Active assignments remain uncertain until reconciled or explicitly retired;
replacement admission is not required to process those recovery controls.
`BuildRequest.max_workers` remains bounded to 1..32.

`Coordinator.worker_lost(worker_id)` fences a confirmed terminal worker instance
in the journal before bounded retry. Worker IDs must identify immutable lifecycle
instances; replacements need fresh IDs and endpoint namespaces. The caller must
verify matching original instance identity, terminal state and a finished
timestamp. For Iris this means the original `TaskAttempt(task_id, attempt_id)`
with matching controller-minted `attempt_uid`, terminal state and non-null
`finished_at`; current-task status, degradation, RPC timeout or a missing UID
does not suffice. Retired IDs remain excluded across coordinator restart and
late completions cannot become authoritative. This guarantees journal fencing:
old private processes may persist under a partition, so it does not promise
physical exactly-once compilation. Independent actions continue normally.

`Coordinator.knows_worker(worker_id)` checks whether an ID belongs to the current
worker mapping, historical assignments or retired instances in this request.
It checks the current journal claim without writes or worker RPCs. Valid unknown
IDs return false; invalid IDs and stale claims raise. Services can use this query
to filter global retirement observations before calling `worker_lost`.

`WorkerExecutor` preserves attempts across client restart using a detached
supervisor, durable launch records, locks and process start identities. A pipe
gate prevents argv execution until child identity is durable. Lost supervisors
cause failure after their action group is stopped; the same attempt is never
executed again. Cancellation kills the action process group. Pending nodes
become cancelled, failed dependency branches become blocked, and independent
branches continue after failures. Retries are bounded by `max_attempts`.

Worker RPCs perform bounded durable metadata work. `submit` persists the full
plan before returning; `poll` reports `RUNNING` while input preparation or output
sealing is pending. Store I/O and workspace cleanup run outside the shared
worker lock, so large SDKs/products do not hold RPCs or independent actions.
One advisory transfer lock owns each attempt's transport. The durable plan,
prepared marker, supervisor launch intent, terminal status and sealed result
drive recovery on worker construction and later polls. Preparation can repeat
after interruption only before launch intent. Interrupted publication reseals
the completed quiescent output, without rerunning argv. Immutable chunk writes
are safe to repeat. This requires the same durable worker root and accessible
store after restart; a permanently lost worker instance uses the coordinator
loss protocol above.

Cancellation writes its durable tombstone immediately. Transfers check it
between bounded blob operations before granting execution or result authority;
an already issued storage operation must return before that transfer can stop.
Actor-owned threads execute the work, while durable records preserve recovery
intent if the actor exits. No in-memory future is required for reconciliation.
Callers must poll to reconcile finished supervisors and receive sealed results.

Completed worker results remain until `Coordinator.acknowledge()` after the
caller consumes a terminal build result. The accepted journal and final result
remain for recovery and idempotency. A failed acknowledgement is retried on
later ticks, including after restart. `WorkerExecutor.read_log(attempt_id,
max_bytes=16384)` returns a bounded stdout/stderr tail, with a 64 KiB per-read
cap. Logs remain until acknowledgement and do not affect identity or publishing.
Worker acknowledgement persists immediately; private workspace/log cleanup is
asynchronous and resumes after restart. Acknowledged attempts report `UNKNOWN`
and cannot be dispatched again, even before cleanup completes.

`Coordinator.cleanup_complete()` checks the current journal claim and returns
true only after request acknowledgement and every attempt's acknowledgement is
durably confirmed. It performs no writes or worker RPCs and raises
`CoordinatorFenced` for a stale coordinator. Services can then retire the request
from active cleanup history; private worker deletion may still be pending.

Output bundles are cache references. Successful execution records do not promise
release durability, cache retention or a TTL. Retrieving an expired manifest or
chunk raises `FileNotFoundError`; corruption raises `ValueError`. The caller must
request a new build rather than treat a missing cached output as published.
