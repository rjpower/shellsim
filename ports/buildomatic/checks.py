"""Compile declared probes or run them in a fresh guest using graph delivery inputs."""

from __future__ import annotations

import fcntl
import json
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from ports._support.acceptance import _fixtures, _native_command, _source, _test_extras, _test_limits
from ports._support.native_artifacts import NativeArtifact, merge_dependency_sysroot
from ports._support.node_build import _native_target
from ports._support.wasm import mark_abi
from ports.buildomatic.artifacts import install, upload, write_reference
from ports.buildomatic.delivery import DeliveryStore, catalog, staged_file
from ports.buildomatic.nodes import Node, Operation
from ports.buildomatic.remote_store import StorageStore
from ports.buildomatic.sdk_artifacts import load_sdk


def probe(node: Node, records: dict, store: StorageStore, cache: Path, work: Path, specification: dict) -> dict:
    components = {name[4:]: record for name, record in records.items() if name.startswith("sdk:")}
    sdk = load_sdk(
        store,
        {"products": components, "target": specification["target"], "dynamic_abi": specification["abi"]},
        cache / "artifacts",
    )
    native = {}
    for name, record in records.items():
        if name.startswith("sdk:"):
            continue
        root = install(store, record["artifact"], cache / "artifacts") / "native"
        if (root / "artifact.json").is_file():
            metadata = json.loads((root / "artifact.json").read_text())
            identity = record["reference"].split("/")[0] + "/" + metadata["inputs"]["recipe"]["name"]
            native[identity] = NativeArtifact(root, metadata)
    merged = work / "sysroot"
    merge_dependency_sysroot(native, native, merged, _native_target(sdk))
    output = work / "product"
    output.mkdir()
    test = node.test
    source, _ = _source(node.port, test["source"])
    command = _native_command(
        sdk,
        merged,
        source,
        output / "check.wasm",
        test.get("link_inputs", []),
        test.get("sdk_link_inputs", []),
        test.get("include_directories", []),
        test.get("link_flags", []),
    )
    if executable := shutil.which("sccache"):
        command = (executable, *command)
    subprocess.run(command, check=True)
    mark_abi(output / "check.wasm", sdk.dynamic_abi.encode())
    return {"artifact": upload(store, output, cache / "archives")}


def stage_asset(store: DeliveryStore, item: dict, root: Path) -> None:
    path = root / item["path"]
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return
        with staged_file(path) as temporary:
            if store.local is not None:
                shutil.copyfile(store.local / item["path"], temporary)
            else:
                store.fs.get_file(store.key(item["path"]), str(temporary))


def check(node: Node, records: dict, store: StorageStore, cache: Path, work: Path) -> dict:
    from shellsim import Environment
    from shellsim._pypi_catalog import Catalog, unpack_files
    from shellsim.pypi import _mount_package_tree

    packages, native, files = [], [], []
    for name, record in records.items():
        if name.startswith("package:"):
            packages.extend(record["delivery"]["wheels"])
            native.extend(record["delivery"]["native"])
            files.extend(record["delivery"].get("assets", []))
    runtime = records.get("runtime", {}).get("runtime")
    assets = cache / "delivery"
    assets.mkdir(exist_ok=True)
    delivery = DeliveryStore(store.prefix)
    items = [*packages, *native, *files, *([runtime] if runtime else [])]
    items.extend(item["precompiled"] for item in tuple(items) if "precompiled" in item)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda item: stage_asset(delivery, item, assets), items))
    test = node.test
    limits = _test_limits(node.port.recipe)
    kwargs = {"limits": limits} if limits is not None else {}
    graph = {"packages": packages, "native": native, "abi": node.port.recipe.get("abi")}
    if runtime:
        graph["runtime"] = {
            **runtime,
            "native_dependencies": [
                item["name"] for item in native if item["destination"].startswith("/usr/lib/python3.13/lib-dynload/")
            ],
        }
    location = work / "catalog.json"
    write_reference(location, catalog(graph, assets.as_uri()))
    installer = Catalog(location, cache_dir=cache / "guest", trusted_precompiled=True)
    if runtime:
        own = next(
            (
                record
                for record in records.values()
                if record.get("reference") == node.port.reference and "delivery" in record
            ),
            {},
        )
        extras = _test_extras(node.port.recipe)
        requirement = node.port.name + ("[" + ",".join(extras) + "]" if extras else "")
        test_packages = [
            wheel["name"]
            for edge in node.port.dependencies
            if edge.kind == "test"
            for wheel in records["package:" + edge.recipe]["delivery"]["wheels"]
        ]
        if own.get("delivery", {}).get("wheels"):
            environment = Environment(
                pypi_packages=[requirement, *test_packages],
                pypi_catalog=location,
                cache_dir=cache / "guest",
                trusted_precompiled=True,
                **kwargs,
            )
        else:
            from shellsim.cpython import CPythonRuntime

            environment = Environment(
                **(kwargs or {"cpu": 10_000_000_000, "memory": 2 * 1024**3, "disk": 512 * 1024**2})
            )
            CPythonRuntime.from_catalog(installer).mount(environment)
    else:
        environment = Environment(**kwargs)
    staging = work / "guest"
    staging.mkdir()
    if not runtime or not own.get("delivery", {}).get("wheels"):
        for item in native:
            installer.register_precompiled(assets / item["path"], item, executable=item["kind"] == "tools")
            path = staging / item["destination"].lstrip("/")
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(assets / item["path"], path)
            path.chmod(0o755 if item["kind"] == "tools" else 0o644)
    for item in files:
        unpack_files(assets / item["path"], staging)
    guest = "/work/check" + {"python": ".py", "shell": ".sh", "native": ".wasm"}[test["kind"]]
    if test["kind"] == "native":
        body = (install(store, records["probe"]["artifact"], cache / "artifacts") / "check.wasm").read_bytes()
    else:
        _, body = _source(node.port, test["script"])
    path = staging / guest.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    path.chmod(0o755 if test["kind"] == "native" else 0o644)
    for name, body in _fixtures(node.port, test.get("files", [])).items():
        path = staging / "work" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _mount_package_tree(
        environment,
        staging,
        "/",
        normalize_modes=True,
        replace_builtin_tools=[item["destination"] for item in native if item["kind"] == "tools"],
    )
    command = {"python": "python ", "shell": "sh ", "native": ""}[test["kind"]] + shlex.quote(guest)
    result = environment.run(" ".join([command, *(shlex.quote(arg) for arg in test.get("args", []))]))
    if result.returncode or result.stop_reason:
        raise RuntimeError(
            f"{node.reference}: {result.returncode}, {result.stop_reason}\n{result.stderr.decode(errors='replace')}"
        )
    return {
        "returncode": result.returncode,
        "usage": asdict(result.usage),
        "stdout": result.stdout.decode(errors="replace")[-8192:],
    }


def execute_check(node: Node, records: dict, store: StorageStore, cache: Path, specification: dict) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="check-", dir=cache) as temporary:
        work = Path(temporary)
        return (
            probe(node, records, store, cache, work, specification)
            if node.operation is Operation.PROBE
            else check(node, records, store, cache, work)
        )
