#!/usr/bin/env python3
"""Inventory Python and environment features in TaskTrove task Parquets.

The TaskTrove schema stores each task as a gzip-compressed tar archive in
``task_binary``.  The small TBLite mirror uses separate base64-encoded
``environment_tar`` and ``tests_tar`` columns.  This tool supports both forms,
never extracts archive members to disk, and emits deterministic JSON to stdout.

Run with an isolated analysis dependency, for example:

    uv run --no-project --with pyarrow \
      python tools/tasktrove_inventory.py tasks.parquet > inventory.json
"""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import sys
import tarfile
from typing import Any, Iterable

MAX_MEMBER_BYTES = 4 * 1024 * 1024
MAX_TASK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PythonFile:
    path: str
    imports: list[str]
    module_accesses: list[str]
    node_types: dict[str, int]
    decorators: list[str]
    test_fixtures: list[str]
    subprocess_calls: list[str]
    syntax_error: str | None


@dataclass(frozen=True)
class TaskInventory:
    task: str
    files: list[str]
    python_files: list[PythonFile]
    imports: list[str]
    module_accesses: list[str]
    node_types: dict[str, int]
    decorators: list[str]
    test_fixtures: list[str]
    subprocess_calls: list[str]
    docker_from: list[str]
    install_commands: list[str]
    runner_commands: list[str]
    archive_errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="TaskTrove-compatible Parquet file or an extracted task-directory root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="inspect only the first N rows (all rows by default)",
    )
    parser.add_argument(
        "--compare-root",
        help="for Parquet input, compare archived files with an extracted task root",
    )
    return parser.parse_args()


def safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    normalized = str(path)
    if normalized in {"", "."}:
        raise ValueError(f"empty archive member path: {name!r}")
    return normalized


def read_tar(blob: bytes, prefix: str = "") -> tuple[dict[str, bytes], list[str]]:
    files: dict[str, bytes] = {}
    errors: list[str] = []
    total = 0
    try:
        archive = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    except (tarfile.TarError, OSError) as error:
        return {}, [f"invalid tar.gz: {error}"]

    with archive:
        for member in archive.getmembers():
            try:
                name = safe_member_name(member.name)
            except ValueError as error:
                errors.append(str(error))
                continue
            if member.isdev() or member.issym() or member.islnk():
                # Links and special files are irrelevant to static source inventory. Refuse to
                # follow them, even though this tool never extracts the archive.
                continue
            if not member.isfile():
                continue
            if member.size > MAX_MEMBER_BYTES:
                errors.append(f"member too large: {name!r} ({member.size} bytes)")
                continue
            total += member.size
            if total > MAX_TASK_BYTES:
                errors.append(f"task archive exceeds {MAX_TASK_BYTES} bytes")
                break
            source = archive.extractfile(member)
            if source is None:
                errors.append(f"could not read member: {name!r}")
                continue
            full_name = str(PurePosixPath(prefix, name)) if prefix else name
            files[full_name] = source.read(MAX_MEMBER_BYTES + 1)
    return files, errors


def decode_base64_tar(value: str, prefix: str) -> tuple[dict[str, bytes], list[str]]:
    try:
        blob = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        return {}, [f"invalid base64 {prefix} archive: {error}"]
    return read_tar(blob, prefix)


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    cursor: ast.AST = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
        return ".".join(reversed(parts))
    return None


def import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.partition(".")[0]
                aliases[bound] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


class FeatureVisitor(ast.NodeVisitor):
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases
        self.imports: set[str] = set()
        # A from-import is itself an API dependency even when the bound name is called or used
        # without an Attribute node (for example ``from dataclasses import dataclass``).
        self.module_accesses: set[str] = {name for name in aliases.values() if "." in name}
        self.node_types: Counter[str] = Counter()
        self.decorators: set[str] = set()
        self.test_fixtures: set[str] = set()
        self.subprocess_calls: set[str] = set()

    def resolve_name(self, node: ast.AST) -> str | None:
        name = dotted_name(node)
        if not name:
            return None
        root, separator, tail = name.partition(".")
        return self.aliases.get(root, root) + separator + tail

    def generic_visit(self, node: ast.AST) -> None:
        self.node_types[type(node).__name__] += 1
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.update(alias.name for alias in node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = self.resolve_name(node)
        root = dotted_name(node).partition(".")[0] if dotted_name(node) else None
        if name and root in self.aliases:
            self.module_accesses.add(name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self.resolve_name(node.func)
        if name and name.startswith("subprocess."):
            argument = ast.unparse(node.args[0]) if node.args else ""
            if len(argument) > 160:
                argument = argument[:157] + "..."
            keywords = ",".join(keyword.arg or "**" for keyword in node.keywords)
            self.subprocess_calls.add(f"{name}({argument}) [{keywords}]")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            name = dotted_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
            if name:
                self.decorators.add(name)
        if node.name.startswith("test_"):
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            self.test_fixtures.update(arg.arg for arg in args if arg.arg not in {"self", "cls"})
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            name = dotted_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
            if name:
                self.decorators.add(name)
        self.generic_visit(node)


def inspect_python(path: str, data: bytes) -> PythonFile:
    text = data.decode("utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=path, type_comments=True)
    except SyntaxError as error:
        location = f"{error.lineno}:{error.offset}" if error.lineno else "unknown"
        return PythonFile(path, [], [], {}, [], [], [], f"{location}: {error.msg}")

    visitor = FeatureVisitor(import_aliases(tree))
    visitor.visit(tree)
    return PythonFile(
        path=path,
        imports=sorted(visitor.imports),
        module_accesses=sorted(visitor.module_accesses),
        node_types=dict(sorted(visitor.node_types.items())),
        decorators=sorted(visitor.decorators),
        test_fixtures=sorted(visitor.test_fixtures),
        subprocess_calls=sorted(visitor.subprocess_calls),
        syntax_error=None,
    )


INSTALL_RE = re.compile(
    r"(?:^|[;&|]\s*)(?P<command>(?:python\S*\s+-m\s+)?(?:pip\S*|uv)\s+[^\n;&|]+)",
    re.MULTILINE,
)
RUNNER_RE = re.compile(r"\b(?:python\S*\s+-m\s+)?(?:pytest|unittest)\b[^\n;&|]*")
FROM_RE = re.compile(r"^\s*FROM\s+([^\s]+)", re.IGNORECASE | re.MULTILINE)
HEREDOC_RE = re.compile(r"<<-?\s*(?P<quote>['\"]?)(?P<marker>[A-Za-z_][A-Za-z0-9_]*)\1")


def text_files(files: dict[str, bytes]) -> Iterable[tuple[str, str]]:
    for path, data in files.items():
        if path.endswith((".py", ".sh", ".txt", ".toml", ".cfg", ".ini")) or PurePosixPath(
            path
        ).name in {"Dockerfile", "requirements.txt"}:
            yield path, data.decode("utf-8", errors="replace")


def load_row_files(row: dict[str, Any]) -> tuple[str, dict[str, bytes], list[str]]:
    errors: list[str] = []
    if "task_binary" in row:
        task = str(row["path"])
        blob = row["task_binary"]
        if not isinstance(blob, bytes):
            blob = bytes(blob)
        files, archive_errors = read_tar(blob)
        return task, files, archive_errors

    task = str(row["task_name"])
    environment, environment_errors = decode_base64_tar(row["environment_tar"], "environment")
    tests, test_errors = decode_base64_tar(row["tests_tar"], "tests")
    errors.extend(environment_errors)
    errors.extend(test_errors)
    files = {**environment, **tests}
    files["tests/test.sh"] = str(row["test_sh"]).encode()
    return task, files, errors


def python_heredocs(path: str, data: bytes) -> Iterable[tuple[str, bytes]]:
    """Yield likely Python heredocs from a shell file without executing the shell."""
    lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
    index = 0
    ordinal = 0
    while index < len(lines):
        header = lines[index]
        match = HEREDOC_RE.search(header)
        index += 1
        if not match:
            continue
        marker = match.group("marker")
        body: list[str] = []
        while index < len(lines) and lines[index].strip() != marker:
            body.append(lines[index])
            index += 1
        if index < len(lines):
            index += 1
        command = header[: match.start()]
        likely_python = bool(re.search(r"(?:^|[\s/])python(?:\d+(?:\.\d+)*)?\b", command))
        writes_python = bool(re.search(r"\.py(?:[\s\"']|$)", header))
        if likely_python or writes_python:
            ordinal += 1
            yield f"{path}::<python-heredoc-{ordinal}>", "".join(body).encode()


def inventory_row(row: dict[str, Any]) -> TaskInventory:
    task, files, errors = load_row_files(row)
    return inventory_files(task, files, errors)


def inventory_directory(root: PurePosixPath) -> list[TaskInventory]:
    from pathlib import Path

    native_root = Path(str(root))
    task_dirs = sorted(path.parent for path in native_root.glob("*/task.toml"))
    inventories: list[TaskInventory] = []
    for task_dir in task_dirs:
        files: dict[str, bytes] = {}
        errors: list[str] = []
        total = 0
        for path in sorted(task_dir.rglob("*")):
            if path.is_symlink() or not path.is_file() or ".git" in path.parts:
                continue
            relative = path.relative_to(task_dir).as_posix()
            size = path.stat().st_size
            if size > MAX_MEMBER_BYTES:
                errors.append(f"member too large: {relative!r} ({size} bytes)")
                continue
            total += size
            if total > MAX_TASK_BYTES:
                errors.append(f"task directory exceeds {MAX_TASK_BYTES} bytes")
                break
            files[relative] = path.read_bytes()
        row_inventory = inventory_files(task_dir.name, files, errors)
        inventories.append(row_inventory)
    return inventories


def inventory_files(task: str, files: dict[str, bytes], errors: list[str]) -> TaskInventory:
    python_sources = [
        *[(path, data) for path, data in sorted(files.items()) if path.endswith(".py")],
        *[
            source
            for path, data in sorted(files.items())
            if path.endswith(".sh")
            for source in python_heredocs(path, data)
        ],
    ]
    python_files = [
        inspect_python(path, data)
        for path, data in python_sources
    ]

    imports = sorted({name for file in python_files for name in file.imports})
    accesses = sorted({name for file in python_files for name in file.module_accesses})
    decorators = sorted({name for file in python_files for name in file.decorators})
    fixtures = sorted({name for file in python_files for name in file.test_fixtures})
    subprocess_calls = sorted({call for file in python_files for call in file.subprocess_calls})
    node_types: Counter[str] = Counter()
    for file in python_files:
        node_types.update(file.node_types)

    docker_from: set[str] = set()
    install_commands: set[str] = set()
    runner_commands: set[str] = set()
    for path, text in text_files(files):
        if PurePosixPath(path).name == "Dockerfile":
            docker_from.update(FROM_RE.findall(text))
        install_commands.update(match.group("command").strip() for match in INSTALL_RE.finditer(text))
        runner_commands.update(match.group(0).strip() for match in RUNNER_RE.finditer(text))

    return TaskInventory(
        task=task,
        files=sorted(files),
        python_files=python_files,
        imports=imports,
        module_accesses=accesses,
        node_types=dict(sorted(node_types.items())),
        decorators=decorators,
        test_fixtures=fixtures,
        subprocess_calls=subprocess_calls,
        docker_from=sorted(docker_from),
        install_commands=sorted(install_commands),
        runner_commands=sorted(runner_commands),
        archive_errors=errors,
    )


def compare_with_root(rows: list[dict[str, Any]], root: str) -> dict[str, Any]:
    from pathlib import Path

    checked = 0
    missing: list[str] = []
    different: list[dict[str, str]] = []
    archive_errors: list[str] = []
    native_root = Path(root)
    for row in rows:
        task, files, errors = load_row_files(row)
        archive_errors.extend(f"{task}: {error}" for error in errors)
        for relative, archived in sorted(files.items()):
            candidate = native_root / task / relative
            label = f"{task}/{relative}"
            if not candidate.is_file():
                missing.append(label)
                continue
            local = candidate.read_bytes()
            checked += 1
            if local != archived:
                different.append(
                    {
                        "path": label,
                        "archive_sha256": hashlib.sha256(archived).hexdigest(),
                        "root_sha256": hashlib.sha256(local).hexdigest(),
                    }
                )
    return {
        "root": str(native_root),
        "checked_files": checked,
        "missing": missing,
        "different": different,
        "archive_errors": archive_errors,
    }


def main() -> int:
    args = parse_args()
    from pathlib import Path

    source = Path(args.source)
    comparison: dict[str, Any] | None = None
    if source.is_dir():
        if args.compare_root:
            raise SystemExit("--compare-root is only valid with Parquet input")
        inventories = inventory_directory(PurePosixPath(source.as_posix()))
        if args.limit is not None:
            inventories = inventories[: args.limit]
    else:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise SystemExit(
                "reading Parquet requires pyarrow; run with `uv run --with pyarrow ...`"
            ) from error
        table = parquet.read_table(source)
        rows = table.to_pylist()
        if args.limit is not None:
            rows = rows[: args.limit]
        inventories = [inventory_row(row) for row in rows]
        if args.compare_root:
            comparison = compare_with_root(rows, args.compare_root)
    report = {
        "schema_version": 1,
        "python_version": sys.version.split()[0],
        "source": str(source),
        "tasks": [asdict(inventory) for inventory in inventories],
    }
    if comparison is not None:
        report["source_comparison"] = comparison
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
