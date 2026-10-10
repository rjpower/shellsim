"""Single-writer CAS journal and incremental DAG scheduling.

One journal owns one idempotent request. Opening an existing journal claims a
new generation, fencing previous coordinators. Dispatch intent and attempt IDs
are durable before RPCs. A failed RPC leaves its worker assignment uncertain;
An explicit UNKNOWN report or confirmed lifecycle terminal identity permits loss
recovery on another worker. RPC errors alone never establish that evidence.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from typing import Mapping

from .contracts import (
    MAX_METADATA_BYTES,
    Attempt,
    AttemptState,
    BuildRequest,
    BuildResult,
    BuildState,
    ConditionalWriteError,
    InputMount,
    NodeResult,
    NodeState,
    Store,
    TreeBundle,
    Worker,
    WorkerState,
    encode,
    name,
    request_from_dict,
    request_id,
)


class IdempotencyConflict(ValueError):
    """The journal's idempotency key or request content differs from submission."""


class CoordinatorFenced(ConditionalWriteError):
    """A newer coordinator owns this journal; no further worker RPCs are allowed."""


def _build_result(journal: dict, request: BuildRequest) -> BuildResult:
    """Validate and aggregate a journal snapshot without changing ownership."""
    if journal["schema"] != 1 or type(journal["generation"]) is not int or journal["generation"] < 1:
        raise ValueError("invalid build journal schema or generation")
    if journal["request_id"] != request_id(request) or type(journal["cancelled"]) is not bool:
        raise ValueError("invalid build journal identity or cancellation state")
    if set(journal["nodes"]) != {action.id for action in request.actions}:
        raise ValueError("invalid build journal nodes")
    nodes = []
    for action in request.actions:
        node = journal["nodes"][action.id]
        attempts = node["attempts"]
        state = NodeState(node["state"])
        bundle = TreeBundle(node["bundle"]) if node["bundle"] is not None else None
        if type(attempts) is not int or not 0 <= attempts <= action.max_attempts:
            raise ValueError("invalid build journal attempt count")
        if state == NodeState.SUCCEEDED and (bundle is None or attempts == 0):
            raise ValueError("invalid successful node result")
        if node["error"] is not None and (not isinstance(node["error"], str) or len(node["error"]) > 4096):
            raise ValueError("invalid node diagnostics")
        nodes.append(NodeResult(action.id, state, attempts, bundle, node["error"]))
    states = {node.state for node in nodes}
    if NodeState.RUNNING in states:
        state = BuildState.RUNNING
    elif NodeState.PENDING in states:
        state = BuildState.PENDING
    elif journal["cancelled"]:
        state = BuildState.CANCELLED
    elif states == {NodeState.SUCCEEDED}:
        state = BuildState.SUCCEEDED
    else:
        state = BuildState.FAILED
    return BuildResult(journal["request_id"], state, tuple(nodes))


def read_build_result(store: Store, journal_key: str) -> BuildResult | None:
    """Read a bounded typed snapshot without CAS, mutation or generation claim.

    None means only that the accepted request journal is absent. Malformed or
    oversized journals raise ValueError. Output bundles remain cache references;
    this read does not fetch blobs or promise that cached outputs still exist.
    """
    current = store.read_journal(journal_key)
    if current is None:
        return None
    if len(current.data) > MAX_METADATA_BYTES:
        raise ValueError("build journal exceeds byte bound")
    try:
        journal = json.loads(current.data)
        request = request_from_dict(journal["request"])
        return _build_result(journal, request)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid build journal") from error


class Coordinator:
    """Drive a request by calling tick; there is no background coordinator thread.

    workers maps immutable instance IDs to durable endpoints. Each endpoint receives
    at most one active node from this request. Removing an endpoint while it has
    an active attempt does not prove loss: restore the endpoint to reconcile it.
    An empty pool admits no execution but still processes recovery controls.
    Use distinct journal keys for independent requests. Acknowledgement releases
    worker results while retaining the request identity and final journal result.
    """

    def __init__(self, store: Store, journal_key: str, workers: Mapping[str, Worker]):
        if not 0 <= len(workers) <= 32:
            raise ValueError("coordinator needs 0..32 workers")
        for key in workers:
            name(key)
        self.store, self.journal_key, self.workers = store, journal_key, dict(workers)
        self._fenced = False
        current = store.read_journal(journal_key)
        self._version = current.version if current else None
        self._journal = json.loads(current.data) if current else None
        self.request = None
        if self._journal is not None:
            if self._journal["schema"] != 1:
                raise ValueError("unsupported coordinator journal")
            self.request = request_from_dict(self._journal["request"])
            self._journal["generation"] += 1
            self._save()

    def _save(self) -> None:
        if self._fenced:
            raise CoordinatorFenced("coordinator has lost its journal claim")
        data = encode(self._journal)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("coordinator journal exceeds byte bound")
        try:
            self._version = self.store.write_journal(self.journal_key, data, self._version)
        except ConditionalWriteError as error:
            self._fenced = True
            raise CoordinatorFenced("coordinator journal revision changed") from error

    def _check(self) -> None:
        if self.request is None:
            raise ValueError("no submitted request")
        if self._fenced:
            raise CoordinatorFenced("coordinator has lost its journal claim")
        current = self.store.read_journal(self.journal_key)
        if current is None or current.version != self._version:
            self._fenced = True
            raise CoordinatorFenced("coordinator journal revision changed")

    def submit(self, request: BuildRequest) -> str:
        identity = request_id(request)
        if self.request is not None:
            self._check()
            if self._journal["request_id"] != identity or self.request.idempotency_key != request.idempotency_key:
                raise IdempotencyConflict("journal already owns a different request")
            return identity
        self.request = request
        self._journal = {
            "schema": 1,
            "generation": 1,
            "request": asdict(request),
            "request_id": identity,
            "cancelled": False,
            "acknowledged": False,
            "lost_workers": [],
            "nodes": {
                action.id: {"state": "pending", "attempts": 0, "bundle": None, "error": None}
                for action in request.actions
            },
            "attempts": {},
        }
        self._save()
        return identity

    def _attempt(self, record: dict) -> Attempt:
        action = next(action for action in self.request.actions if action.id == record["action_id"])
        inputs = action.inputs + tuple(
            InputMount(dep, TreeBundle(self._journal["nodes"][dep]["bundle"])) for dep in action.dependencies
        )
        return Attempt(record["id"], self._journal["request_id"], action, inputs, record["generation"])

    def _ack_workers(self) -> None:
        for record in self._journal["attempts"].values():
            if record["active"] or not record["ack_pending"]:
                continue
            if record["worker"] in self._journal.get("lost_workers", ()):
                record["ack_pending"] = False
                self._save()
                continue
            worker = self.workers.get(record["worker"])
            if worker is None:
                continue
            self._save()
            try:
                worker.acknowledge(record["id"])
            except Exception:
                continue
            record["ack_pending"] = False
            self._save()

    def _finish(self, record: dict, state: AttemptState, bundle: TreeBundle | None, error: str | None) -> None:
        if state == AttemptState.SUCCEEDED and not isinstance(bundle, TreeBundle):
            raise ValueError("successful attempt has no output bundle")
        if error is not None and not isinstance(error, str):
            raise ValueError("attempt diagnostics must be text")
        node = self._journal["nodes"][record["action_id"]]
        record.update(active=False, ack_pending=record["worker"] not in self._journal.get("lost_workers", ()))
        action = next(action for action in self.request.actions if action.id == record["action_id"])
        node["error"] = error[:4096] if error else None
        if self._journal["cancelled"] or state == AttemptState.CANCELLED:
            node["state"] = "cancelled"
        elif state == AttemptState.SUCCEEDED:
            node.update(state="succeeded", bundle=bundle.digest, error=None)
        elif node["attempts"] < action.max_attempts:
            node["state"] = "pending"
        else:
            node["state"] = "failed"
        self._save()

    def tick(self) -> BuildResult:
        """Reconcile attempts, then immediately dispatch newly ready nodes.

        Any worker RPC exception retains the exact attempt and assignment. This
        includes uncertain submission and cancellation, preventing duplicate work.
        """
        self._check()
        for record in list(self._journal["attempts"].values()):
            if not record["active"]:
                continue
            if record["worker"] in self._journal.get("lost_workers", ()):
                self._finish(record, AttemptState.FAILED, None, "worker instance terminated")
                continue
            worker = self.workers.get(record["worker"])
            if worker is None:
                continue
            self._save()
            try:
                if self._journal["cancelled"]:
                    worker.cancel(record["id"])
                report = worker.poll(record["id"])
            except Exception:
                continue
            if report.state == WorkerState.COMPLETED:
                result = report.result
                if result is None or result.attempt_id != record["id"] or not isinstance(result.state, AttemptState):
                    raise ValueError("worker returned an invalid attempt result")
                self._finish(record, result.state, result.bundle, result.error)
            elif report.state == WorkerState.UNKNOWN:
                if self._journal["cancelled"] or record["dispatched"]:
                    # Tombstones fence late RPCs before another attempt can run.
                    self._save()
                    try:
                        worker.cancel(record["id"])
                    except Exception:
                        continue
                    self._finish(record, AttemptState.FAILED, None, "worker lost attempt")
                else:
                    self._save()
                    try:
                        worker.submit(self._attempt(record))
                    except Exception:
                        continue
                    record["dispatched"] = True
                    self._save()
            elif report.state != WorkerState.RUNNING:
                raise ValueError("worker returned an invalid state")

        nodes = self._journal["nodes"]
        changed = True
        while changed:
            changed = False
            for action in self.request.actions:
                node = nodes[action.id]
                if node["state"] == "pending" and any(
                    nodes[dep]["state"] in ("failed", "blocked", "cancelled") for dep in action.dependencies
                ):
                    node.update(state="blocked", error="dependency did not succeed")
                    changed = True
            if changed:
                self._save()

        active = [record for record in self._journal["attempts"].values() if record["active"]]
        occupied = {record["worker"] for record in active} | set(self._journal.get("lost_workers", ()))
        available = [key for key in self.workers if key not in occupied]
        slots = max(0, self.request.max_workers - len(active))
        if not self._journal["cancelled"]:
            for action in self.request.actions:
                node = nodes[action.id]
                if not slots or not available:
                    break
                if node["state"] != "pending" or any(nodes[dep]["state"] != "succeeded" for dep in action.dependencies):
                    continue
                record = {
                    "id": uuid.uuid4().hex,
                    "action_id": action.id,
                    "worker": available.pop(0),
                    "generation": self._journal["generation"],
                    "active": True,
                    "dispatched": False,
                    "ack_pending": False,
                }
                self._journal["attempts"][record["id"]] = record
                node["state"] = "running"
                node["attempts"] += 1
                slots -= 1
                self._save()
                try:
                    self.workers[record["worker"]].submit(self._attempt(record))
                except Exception:
                    continue
                record["dispatched"] = True
                self._save()
        # Worker outputs stay retained until the caller acknowledges the request.
        if self._journal["acknowledged"]:
            self._ack_workers()
        return self.result()

    def worker_lost(self, worker_id: str) -> BuildResult:
        """Fence a confirmed terminal instance durably before bounded retry.

        The caller must prove terminal lifecycle identity for this exact immutable
        worker ID. RPC errors, missing endpoints and degradation are insufficient.
        Replacement instances require fresh IDs/namespaces. Retired IDs stay
        fenced after coordinator restart and late results cannot become authoritative.
        This fences journal authority; partitioned private processes may persist.
        """
        self._check()
        name(worker_id)
        known = (
            set(self.workers)
            | {record["worker"] for record in self._journal["attempts"].values()}
            | set(self._journal.get("lost_workers", ()))
        )
        if worker_id not in known:
            raise ValueError("worker identity was never assigned or registered")
        retired = self._journal.setdefault("lost_workers", [])
        if worker_id not in retired:
            retired.append(worker_id)
            self._save()
        return self.tick()

    def cancel(self) -> BuildResult:
        """Persist cancellation before delivering it; tick retries uncertain RPCs."""
        self._check()
        if self.result().state in (BuildState.SUCCEEDED, BuildState.FAILED, BuildState.CANCELLED):
            return self.result()
        self._journal["cancelled"] = True
        for node in self._journal["nodes"].values():
            if node["state"] == "pending":
                node["state"] = "cancelled"
        self._save()
        return self.tick()

    def result(self) -> BuildResult:
        self._check()
        return _build_result(self._journal, self.request)

    def acknowledge(self) -> None:
        """Release terminal worker results after the caller durably consumes them."""
        if self.result().state in (BuildState.PENDING, BuildState.RUNNING):
            raise ValueError("cannot acknowledge an active request")
        self._journal["acknowledged"] = True
        self._save()
        self._ack_workers()

    def cleanup_complete(self) -> bool:
        """Confirm durable request acknowledgement and all worker acknowledgements.

        Check the current journal claim without mutation or worker RPCs. True
        permits removing this request from the service's active cleanup history;
        it does not wait for asynchronous private workspace deletion.
        """
        self._check()
        return self._journal["acknowledged"] and all(
            not record["ack_pending"] for record in self._journal["attempts"].values()
        )
