#!/usr/bin/env python3
"""Build a deterministic stdlib/API usage matrix from an extracted TaskTrove tree.

The tool intentionally uses only the Python standard library.  It reads ``.py`` files and
Python heredocs in ``solution/*.sh`` without executing either.  The JSON report is useful for
automation; ``--format tsv`` is convenient for review and spreadsheet tooling.

Example::

    python3 tools/tasktrove_stdlib_matrix.py /tmp/OpenThoughts-TBLite \
      --format json > /tmp/stdlib.json
    python3 tools/tasktrove_stdlib_matrix.py /tmp/OpenThoughts-TBLite \
      --format tsv > /tmp/stdlib.tsv

The output is sorted by module and API, so rerunning against the same tree is byte-for-byte
reproducible (apart from the report's source path).
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Iterable


MODULES = (
    "sys", "os", "collections", "itertools", "heapq", "bisect", "math", "string",
    "json", "re", "functools", "dataclasses", "typing", "enum", "argparse", "subprocess",
    "pytest", "unittest",
)

# A default placement is deliberately conservative: a VFS/process boundary must be explicit,
# while deterministic, side-effect-free modules can be implemented as ordinary shims.
CLASSIFICATION = {
    "sys": "safe native", "os": "VFS wrapper", "collections": "pure shim",
    "itertools": "pure shim", "heapq": "pure shim", "bisect": "pure shim",
    "math": "safe native", "string": "pure shim", "json": "pure shim",
    "re": "pure shim", "functools": "pure shim", "dataclasses": "pure shim",
    "typing": "pure shim", "enum": "pure shim", "argparse": "pure shim",
    "subprocess": "VFS wrapper", "pytest": "pure shim", "unittest": "pure shim",
}

STAGE = {
    "sys": 1, "os": 1, "json": 1, "collections": 1, "typing": 1,
    "math": 2, "string": 2, "re": 2, "functools": 2, "itertools": 2,
    "heapq": 2, "bisect": 2, "dataclasses": 2, "enum": 2, "argparse": 2,
    "subprocess": 3, "pytest": 3, "unittest": 3,
}

NOTES = {
    "sys": "argv, path, stdin/stdout, exit; expose immutable snapshots where possible",
    "os": "route filesystem operations through VFS and reject host escape",
    "collections": "defaultdict, deque, Counter are the observed high-value subset",
    "itertools": "combinatorics and iterator helpers; keep iteration metered",
    "heapq": "heappush/heappop over mutable lists",
    "bisect": "reviewer-requested; no TaskTrove occurrence in this sample",
    "math": "deterministic numeric helpers; define non-finite behavior explicitly",
    "string": "constants first (ascii_lowercase observed)",
    "json": "loads/dumps plus file load/dump; bound recursion and output size",
    "re": "compile/search/findall/sub and flags; bound pattern/input work",
    "functools": "reviewer-requested; add wraps/partial/lru_cache only with resource limits",
    "dataclasses": "dataclass decorator and fields; generated methods must remain deterministic",
    "typing": "runtime no-op marker objects sufficient for annotations",
    "enum": "Enum values and member/name/value behavior",
    "argparse": "ArgumentParser/Namespace for deterministic python -c wrappers",
    "subprocess": "only synchronous shellsim/python wrappers; reject arbitrary host processes",
    "pytest": "native runner facade, fixtures and common flags; do not embed upstream pytest",
    "unittest": "separate runner facade; absent from TBLite, so use reviewer conformance tests",
}

UNSUPPORTED_APIS = {
    "os.getpgid", "os.kill", "os.killpg", "os.setsid", "os.system", "pytest.mark.asyncio",
}

HEREDOC_RE = re.compile(r"<<-?\s*(?P<quote>['\"]?)(?P<marker>[A-Za-z_][A-Za-z0-9_]*)\1")


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def python_heredocs(path: str, text: str) -> Iterable[tuple[str, str]]:
    lines = text.splitlines(keepends=True)
    index = ordinal = 0
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
        command = header[:match.start()]
        if re.search(r"(?:^|[\s/])python(?:\d+(?:\.\d+)*)?\b", command) or re.search(
            r"\.py(?:[\s\"']|$)", header
        ):
            ordinal += 1
            yield f"{path}::<python-heredoc-{ordinal}>", "".join(body)


def source_files(root: Path) -> Iterable[tuple[str, str, str]]:
    """Yield task, provenance path, source in stable order."""
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = sorted(path for path in task_dir.rglob("*") if path.is_file() and not path.is_symlink())
        for path in paths:
            relative = path.relative_to(task_dir).as_posix()
            if relative.endswith(".py"):
                yield task_dir.name, relative, path.read_text(encoding="utf-8", errors="replace")
            elif relative.endswith(".sh"):
                yield from ((task_dir.name, name, body) for name, body in python_heredocs(relative, path.read_text(encoding="utf-8", errors="replace")))


def task_names(root: Path) -> list[str]:
    return sorted(path.parent.name for path in root.glob("*/task.toml"))


def classification(module: str, api: str | None = None) -> str:
    if api in UNSUPPORTED_APIS:
        return "unsupported"
    return CLASSIFICATION[module]


class UsageVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.imported_modules: set[str] = set()
        self.accesses: Counter[str] = Counter()
        self.calls: Counter[str] = Counter()
        self.imports: Counter[str] = Counter()

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            root = item.name.partition(".")[0]
            if root not in MODULES:
                continue
            bound = item.asname or root
            self.aliases[bound] = item.name
            self.imported_modules.add(root)
            self.imports[item.name] += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.partition(".")[0]
        if root in MODULES:
            self.imported_modules.add(root)
            self.imports[module] += 1
            for item in node.names:
                if item.name != "*":
                    self.aliases[item.asname or item.name] = f"{module}.{item.name}"
        self.generic_visit(node)

    def resolve(self, node: ast.AST) -> str | None:
        name = dotted_name(node)
        if name is None:
            if isinstance(node, ast.Name):
                return self.aliases.get(node.id)
            return None
        root, _, tail = name.partition(".")
        canonical = self.aliases.get(root)
        if canonical is None and root in MODULES:
            canonical = root
        if canonical is None:
            return None
        # Attribute chains beyond the documented submodules are usually a local object that
        # happens to have the same spelling as an imported module (for example
        # ``subprocess.time.time``).  Keep the matrix about actual module APIs, while allowing
        # the observed nested namespaces such as os.path and pytest.mark.
        if tail and tail.count(".") >= 1 and f"{canonical}.{tail.partition('.')[0]}" not in {
            "os.path", "sys.path", "pytest.mark"
        }:
            return None
        return canonical + (f".{tail}" if tail else "")

    def record(self, node: ast.AST, calls: bool = False) -> None:
        name = self.resolve(node)
        if name is None:
            return
        root = name.partition(".")[0]
        if root not in MODULES:
            return
        self.accesses[name] += 1
        if calls:
            self.calls[name] += 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.record(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Names imported with ``from module import symbol`` are API accesses too.
        if isinstance(node.ctx, ast.Load) and node.id in self.aliases:
            self.record(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.record(node.func, calls=True)
        # The function expression is already recorded above; visiting it again would count
        # every call twice in ``accesses``.  Its arguments still contain ordinary API uses.
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)


def inspect(task: str, source: str, path: str, rows: dict[tuple[str, str], dict[str, object]]) -> None:
    try:
        tree = ast.parse(source, filename=f"{task}/{path}", type_comments=True)
    except SyntaxError:
        return
    visitor = UsageVisitor()
    visitor.visit(tree)
    for module in sorted(visitor.imported_modules):
        key = (module, "__module__")
        row = rows.setdefault(key, {"module": module, "api": "__module__", "accesses": 0, "calls": 0, "imports": 0, "tasks": set(), "sources": set()})
        row["imports"] = int(row["imports"]) + sum(count for name, count in visitor.imports.items() if name.partition(".")[0] == module)
        row["tasks"].add(task)  # type: ignore[union-attr]
        row["sources"].add(f"{task}/{path}")  # type: ignore[union-attr]
    for api, count in visitor.accesses.items():
        module = api.partition(".")[0]
        if module not in MODULES or api == module:
            continue
        key = (module, api)
        row = rows.setdefault(key, {"module": module, "api": api, "accesses": 0, "calls": 0, "imports": 0, "tasks": set(), "sources": set()})
        row["accesses"] = int(row["accesses"]) + count
        row["calls"] = int(row["calls"]) + visitor.calls.get(api, 0)
        row["tasks"].add(task)  # type: ignore[union-attr]
        row["sources"].add(f"{task}/{path}")  # type: ignore[union-attr]


def report(root: Path) -> dict[str, object]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for task, path, source in source_files(root):
        inspect(task, source, path, rows)
    modules: list[dict[str, object]] = []
    apis: list[dict[str, object]] = []
    # Include reviewer-requested modules even when the sample has no import.  This makes an
    # absence visible instead of silently dropping bisect, functools, or unittest from the plan.
    for module in MODULES:
        rows.setdefault((module, "__module__"), {"module": module, "api": "__module__", "accesses": 0, "calls": 0, "imports": 0, "tasks": set(), "sources": set()})
    for (module, api), row in sorted(rows.items()):
        tasks = sorted(row.pop("tasks"))  # type: ignore[arg-type]
        sources = sorted(row.pop("sources"))  # type: ignore[arg-type]
        output = {
            **row,
            "task_count": len(tasks),
            "source_count": len(sources),
            "tasks": tasks,
            "sources": sources,
        }
        if api == "__module__":
            modules.append({
                "module": module,
                "task_count": len(tasks),
                "source_count": len(sources),
                "import_count": output["imports"],
                "classification": classification(module),
                "stage": STAGE[module],
                "notes": NOTES[module],
                "tasks": tasks,
                "sources": sources,
            })
        else:
            apis.append({
                **output,
                "classification": classification(module, api),
                "stage": STAGE[module],
            })
    return {
        "schema_version": 1,
        "source": str(root),
        "task_count": len(task_names(root)),
        "observed_task_count": len({task for row in modules for task in row["tasks"]}),
        "modules": modules,
        "apis": apis,
    }


def compact_result(result: dict[str, object]) -> dict[str, object]:
    """Retain counts and representative provenance while bounding repeated path text.

    The un-compacted JSON remains available for audits.  Checked-in reports use this form so
    review diffs stay readable; ``task_examples`` and ``source_examples`` still make every row
    traceable back to the corpus.
    """
    compact: dict[str, object] = {key: value for key, value in result.items() if key not in {"modules", "apis"}}
    for key in ("modules", "apis"):
        entries = []
        for row in result[key]:  # type: ignore[index]
            row = dict(row)
            tasks = row.pop("tasks", [])
            sources = row.pop("sources", [])
            row["task_examples"] = tasks[:5]
            row["source_examples"] = sources[:3]
            entries.append(row)
        compact[key] = entries
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="extracted TaskTrove task-directory root")
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--compact", action="store_true", help="keep counts plus five task and three source examples per row")
    args = parser.parse_args()
    result = report(args.root)
    if args.compact:
        result = compact_result(result)
    if args.format == "json":
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print("kind\tmodule\tapi\ttask_count\tsource_count\taccesses\tcalls\timports\tclassification\tstage\ttasks\tsources")
        for row in result["modules"] + result["apis"]:  # type: ignore[operator]
            kind = "module" if "api" not in row else "api"
            values = [kind]
            source_key = "source_examples" if args.compact else "sources"
            for key in ("module", "api", "task_count", "source_count", "accesses", "calls", "imports", "classification", "stage", "tasks", source_key):
                value = row.get(key, "")
                values.append(";".join(value) if key in {"tasks", "sources"} else str(value))
            print("\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
