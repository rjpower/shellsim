"""Run ordinary Buildomatic actions on this machine using the core ready queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from ports.buildomatic import BuildRequest, Coordinator, LocalStore, ResourceLimits, TreeBundle, WorkerExecutor
from ports.buildomatic.artifacts import resolve
from ports.buildomatic.contracts import Store
from ports.buildomatic.transfers import upload_bundles

_DEFAULT_LIMITS = ResourceLimits(memory_bytes=16 * 1024**3)


class LocalBackend:
    """Local placement for the same requests and worker protocol used by Iris."""

    def __init__(self, store: Store, root: Path, *, slots: int = 1, limits: ResourceLimits = _DEFAULT_LIMITS):
        self.store, self.root, self.slots, self.limits = store, root, slots, limits
        self.builds: dict[str, Coordinator] = {}

    def resolve_artifact(self, key: str) -> dict | None:
        return resolve(self.store, key)

    def upload_bundles(self, source: LocalStore, bundles: Sequence[TreeBundle]) -> None:
        from contextlib import nullcontext

        upload_bundles(source, bundles, factory=lambda: nullcontext(self.store))

    def submit(self, request: BuildRequest) -> str:
        workers = {
            "local-" + str(slot): WorkerExecutor(
                self.store,
                self.root / ("worker-" + str(slot)),
                self.limits,
                environment={
                    "BUILDOMATIC_CACHE": str((self.root / "cache").resolve()),
                    "SHELLSIM_WASMTIME_CACHE_DIR": str((self.root / "wasmtime").resolve()),
                },
            )
            for slot in range(self.slots)
        }
        # Coordination is private to this submission; product identities are shared.
        build_id = uuid4().hex
        coordinator = Coordinator(self.store, "local/builds/" + build_id, workers)
        coordinator.submit(request)
        self.builds[build_id] = coordinator
        return build_id

    def get(self, build_id: str):
        return self.builds[build_id].tick()

    def cancel(self, build_id: str):
        return self.builds[build_id].cancel()

    def get_logs(self, build_id: str) -> dict:
        coordinator = self.builds[build_id]
        journal = json.loads(self.store.read_journal(coordinator.journal_key).data)
        tails = {}
        attempts = list(journal["attempts"].values())
        failed = {action for action, node in journal["nodes"].items() if node["state"] == "failed"}
        if failed:
            attempts = [attempt for attempt in attempts if attempt["action_id"] in failed]
        for attempt in attempts[-4:]:
            tails[attempt["id"]] = (
                coordinator.workers[attempt["worker"]].read_log(attempt["id"]).decode(errors="replace")
            )
        return tails

    def acknowledge(self, build_id: str) -> None:
        self.builds[build_id].acknowledge()
