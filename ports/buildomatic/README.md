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
Attempt IDs and assigned workers are durable before dispatch. Worker RPC errors
retain assignments; they never imply loss. Only explicit `UNKNOWN` permits a
bounded retry, after cancellation tombstones fence delayed dispatch. A worker
endpoint removed from the mapping remains uncertain until restored.

`WorkerExecutor` preserves attempts across client restart using a detached
supervisor, durable launch records, locks and process start identities. A pipe
gate prevents argv execution until child identity is durable. Lost supervisors
cause failure after their action group is stopped; the same attempt is never
executed again. Cancellation kills the action process group. Pending nodes
become cancelled, failed dependency branches become blocked, and independent
branches continue after failures. Retries are bounded by `max_attempts`.

Completed worker results remain until `Coordinator.acknowledge()` after the
caller consumes a terminal build result. The accepted journal and final result
remain for recovery and idempotency. A failed acknowledgement is retried on
later ticks, including after restart. `WorkerExecutor.read_log(attempt_id,
max_bytes=16384)` returns a bounded stdout/stderr tail, with a 64 KiB per-read
cap. Logs remain until acknowledgement and do not affect identity or publishing.

Output bundles are cache references. Successful execution records do not promise
release durability, cache retention or a TTL. Retrieving an expired manifest or
chunk raises `FileNotFoundError`; corruption raises `ValueError`. The caller must
request a new build rather than treat a missing cached output as published.
