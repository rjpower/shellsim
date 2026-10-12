"""Ports commands share one planner and worker; backends choose execution placement."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ports", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ports", type=Path, default=Path(__file__).resolve().parents[1])
    common.add_argument(
        "--state", type=Path, default=Path("build/ports"), help="graph.json, results.json and client state"
    )
    common.add_argument("--store", help="canonical local directory or S3/GCS prefix")
    common.add_argument("--sdk", default="default")
    common.add_argument("--backend", choices=("local", "iris"), default="local")
    common.add_argument("--iris-service", type=Path)
    common.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    common.add_argument("--max-workers", type=int, default=1)
    common.add_argument("--offline", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "export", "check"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument("recipes", nargs="+") if name == "build" else command.add_argument("recipes", nargs="*")
        if name == "export":
            command.add_argument("--url-prefix", required=True)
            command.add_argument("--output", help="catalog directory or store prefix")
            command.add_argument("--copy-assets", action="store_true")
            command.add_argument("--simple", action="store_true")
    args = parser.parse_args(arguments)
    if args.jobs < 1 or not 1 <= args.max_workers <= 32:
        parser.error("jobs must be positive; max-workers requires 1..32")
    saved = (
        json.loads((args.state / "results.json").read_text())
        if not args.recipes and (args.state / "results.json").is_file()
        else {}
    )
    requests = args.recipes or saved.get("roots", [])
    if not requests:
        parser.error("specify recipes or build them first")
    from ports.buildomatic.backends.local import LocalBackend
    from ports.buildomatic.nodes import Operation
    from ports.buildomatic.remote_store import StorageStore
    from ports.buildomatic.workflow import run

    connection = (
        json.loads((args.iris_service or args.state / "service.json").read_text()) if args.backend == "iris" else None
    )
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
    if args.backend == "iris":
        from ports.buildomatic.backends.iris import backend_context_factory

        context = backend_context_factory(
            job_id=connection["job_id"],
            cluster_name=connection["cluster_name"],
            controller_url=connection["controller_url"],
            workspace=Path(connection["workspace"]),
            prefix=connection["prefix"],
            cache_prefix=connection.get("cache_prefix"),
        )()
    else:
        context = nullcontext(LocalBackend(store, args.state / "execution", slots=args.max_workers))
    with context as backend:
        result = run(
            args.ports,
            requests,
            store,
            args.state,
            backend,
            operation={"build": Operation.BUILD, "export": Operation.DELIVERY, "check": Operation.CHECK}[args.command],
            sdk_name=saved.get("sdk_name", args.sdk),
            jobs=args.jobs,
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
