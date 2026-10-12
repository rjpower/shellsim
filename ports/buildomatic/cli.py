"""Ports commands share one planner and worker; backends choose execution placement."""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence
from uuid import uuid4

logger = logging.getLogger(__name__)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ports", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ports", type=Path, default=Path(__file__).resolve().parents[1])
    common.add_argument(
        "--state", type=Path, default=Path("build/ports"), help="graph.json, results.json and client state"
    )
    common.add_argument("--store", help="canonical local directory or S3/GCS prefix")
    common.add_argument("--sdk")
    common.add_argument("--backend", choices=("local", "iris"), default="local")
    common.add_argument("--iris-service", type=Path, help="use an existing pool instead of launching one")
    common.add_argument("--worker-cpus", type=int, help="CPUs per new Iris worker (default: 64)")
    common.add_argument("--worker-memory-gb", type=int, help="RAM per new Iris worker (default: 32 GiB)")
    common.add_argument("--worker-disk-gb", type=int, help="disk per new Iris worker (default: 100 GiB)")
    common.add_argument("--iris-image", help="image for new Iris workers (default: published Buildomatic image)")
    common.add_argument("--jobs", type=int, help="compiler parallelism (default: execution worker CPUs)")
    common.add_argument("--max-workers", type=int, default=1)
    common.add_argument("--offline", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    parser.set_defaults(all=False)
    for name in ("build", "export", "check"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument("recipes", nargs="*")
        if name == "build":
            command.add_argument("--all", action="store_true", help="build every guest port, including recipe variants")
        if name == "export":
            command.add_argument("--url-prefix", required=True)
            command.add_argument("--output", help="catalog directory or store prefix")
            command.add_argument("--copy-assets", action="store_true")
            command.add_argument("--simple", action="store_true")
    args = parser.parse_args(arguments)
    from rigging.log_setup import configure_logging

    from ports._support.graph import all_ports

    configure_logging()
    if args.all and args.recipes:
        parser.error("choose --all or explicit recipes")
    if args.command == "build" and not args.all and not args.recipes:
        parser.error("specify recipes or --all")
    if (args.jobs is not None and args.jobs < 1) or not 1 <= args.max_workers <= 32:
        parser.error("jobs must be positive; max-workers requires 1..32")
    worker_options = (args.worker_cpus, args.worker_memory_gb, args.worker_disk_gb, args.iris_image)
    if any(value is not None for value in worker_options) and (args.backend != "iris" or args.iris_service):
        parser.error("worker resource and image options require a new Iris pool")
    if any(value is not None and value < 1 for value in worker_options[:3]):
        parser.error("worker resources must be positive")
    if args.iris_service and args.backend != "iris":
        parser.error("--iris-service requires --backend iris")
    saved = (
        json.loads((args.state / "results.json").read_text())
        if args.command != "build" and not args.recipes and (args.state / "results.json").is_file()
        else {}
    )
    requests = all_ports(args.ports) if args.all else args.recipes or saved.get("roots", [])
    if not requests:
        parser.error("specify recipes or build them first")
    from ports.buildomatic.backends.local import LocalBackend
    from ports.buildomatic.nodes import Operation
    from ports.buildomatic.remote_store import StorageStore
    from ports.buildomatic.workflow import run

    connection = json.loads(args.iris_service.read_text()) if args.iris_service else None
    prefix = (
        args.store
        or saved.get("prefix")
        or (
            (connection.get("cache_prefix") or connection["prefix"])
            if connection
            else str((args.state / "store").resolve())
        )
    )
    store = StorageStore(prefix)
    worker_cpus = args.worker_cpus or 64
    jobs = args.jobs or (worker_cpus if args.backend == "iris" else os.cpu_count() or 1)
    if args.backend == "iris":
        from rigging.filesystem.storage_path import StoragePath

        from ports.buildomatic.backends.iris import (
            DEFAULT_TASK_IMAGE,
            AutomaticIrisBackend,
            IrisConfig,
            backend_context_factory,
        )

        if StoragePath(store.prefix).is_local:
            parser.error("Iris workers require a cloud --store (S3 or GCS)")
        if connection:
            logger.info("Connecting to Iris worker pool %s", connection["job_id"])
            context = backend_context_factory(
                job_id=connection["job_id"],
                cluster_name=connection["cluster_name"],
                controller_url=connection["controller_url"],
                workspace=Path(connection["workspace"]),
                prefix=connection["prefix"],
                cache_prefix=connection.get("cache_prefix"),
            )()
        else:
            service_id = "ports-" + uuid4().hex[:16]
            config = IrisConfig(
                service_id=service_id,
                prefix=store.prefix.rstrip("/") + "/services/" + service_id,
                cache_prefix=store.prefix,
                workers=args.max_workers,
                worker_cpu=worker_cpus,
                worker_memory_bytes=(args.worker_memory_gb or 32) * 1024**3,
                worker_disk_bytes=(args.worker_disk_gb or 100) * 1024**3,
                worker_output_bytes=(args.worker_disk_gb or 100) * 1024**3,
                task_image=args.iris_image or DEFAULT_TASK_IMAGE,
                compiler_cache=True,
                setup_scripts=("uv sync --no-dev --no-install-project", "uv pip install /opt/buildomatic/wheels/*.whl"),
            )
            logger.info("Iris will allocate up to %d workers on cache misses", args.max_workers)
            context = AutomaticIrisBackend(config, store, args.ports.resolve().parent)
    else:
        logger.info("Using %d local build workers, %d compiler jobs per port", args.max_workers, jobs)
        context = nullcontext(LocalBackend(store, args.state / "execution", slots=args.max_workers))
    with context as backend:
        result = run(
            args.ports,
            requests,
            store,
            args.state,
            backend,
            operation={"build": Operation.BUILD, "export": Operation.DELIVERY, "check": Operation.CHECK}[args.command],
            sdk_name=args.sdk or saved.get("sdk_name", "default"),
            jobs=jobs,
            max_workers=args.max_workers,
            offline=args.offline,
        )
    if args.command == "export":
        from ports.buildomatic.delivery import export

        print(
            json.dumps(
                export(
                    args.state,
                    args.output or args.state / "export",
                    args.url_prefix,
                    copy_assets=args.copy_assets,
                    simple=args.simple,
                )
            )
        )
    else:
        print(json.dumps({"command": args.command, "records": len(result.records), "state": str(args.state)}))
