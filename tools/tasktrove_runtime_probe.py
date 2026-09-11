#!/usr/bin/env python3
"""Probe unchanged TaskTrove Python with the shellsim Python runner.

This is a compatibility probe, not a TaskTrove solver. It loads Python payloads from each
reference solution and asks shellsim to collect each task's verifier suite. Results record only
the first blocker reached for each source or suite. Host CPython is used only by this analysis
harness to enumerate inputs and invoke the capability-free shellsim runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable

from tasktrove_inventory import inspect_python, python_heredocs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted TaskTrove task-directory root")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("target/release/shellsim-python"),
        help="shellsim-python executable (default: target/release/shellsim-python)",
    )
    parser.add_argument("--limit", type=int, help="probe only the first N tasks")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="host timeout per invocation in seconds (default: 10)",
    )
    return parser.parse_args()


def source_error_line(diagnostic: str, source: bytes) -> str:
    match = re.search(r" at line (\d+), column", diagnostic)
    if not match:
        return ""
    lines = source.decode("utf-8", errors="replace").splitlines()
    line_number = int(match.group(1))
    return lines[line_number - 1] if 0 < line_number <= len(lines) else ""


def classify(diagnostic: str, unsupported: list[str], source: bytes | None = None) -> str:
    """Map the first shellsim diagnostic to a stable feature-sized category."""
    text = "\n".join([diagnostic, *unsupported])
    module = re.search(r"no module named [\"']([^\"']+)", text)
    if module:
        return f"module:{module.group(1).split('.')[0]}"
    option = re.search(r"pytest option ([^\s]+)", text)
    if option:
        return f"pytest-option:{option.group(1)}"
    if "fixture arguments" in text:
        return "pytest:fixtures"
    if "decorators are unsupported" in text:
        return "pytest:decorators"

    line = source_error_line(text, source) if source is not None else ""
    decoded = source.decode("utf-8", errors="replace") if source is not None else ""
    if ('"""' in line or "'''" in line) and (
        "unterminated string literal" in text
        or "expected a newline or ';' after statement" in text
    ):
        return "syntax:triple-quoted-strings"
    if "unexpected character '\\\\'" in text:
        return "syntax:explicit-line-continuation"
    if "f-string" in text or re.search(r"\b(?:rf|fr)[\"']", line, re.IGNORECASE):
        return "syntax:f-string-formatting"
    if "expected a module name after 'from'" in text:
        return "syntax:relative-import"
    if line.lstrip().startswith("async def ") or (
        "decorators may only be applied" in text
        and re.search(r"^\s*async\s+def\b", decoded, re.MULTILINE)
    ):
        return "syntax:async-functions"
    if any(character in line for character in (" & ", " | ", " ^ ")) or any(
        f"unexpected character '{character}'" in text for character in "&|^"
    ):
        return "syntax:bitwise-operators"
    if re.search(r"\bif\b.+\belse\b", line):
        return "syntax:conditional-expressions"
    if re.search(r"\[[^\]]*:[^\]]*\]", line):
        return "syntax:slices"
    if "expected an indented suite" in text and line.lstrip().startswith("#"):
        return "syntax:comment-first-suites"
    if "expected" in text or "unsupported" in text or "not implemented" in text:
        return f"language-api:{text.strip().splitlines()[0][:100]}"
    if "AssertionError" in text or "FAILED" in text:
        return "test-failure"
    if not text.strip():
        return "nonzero-without-diagnostic"
    return f"runtime:{text.strip().splitlines()[0][:100]}"


def invoke(runner: Path, arguments: list[str], timeout: float) -> tuple[bool, str, list[str]]:
    try:
        result = subprocess.run(
            [str(runner), "--json", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "host probe timeout", []
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, result.stderr or result.stdout, []
    return (
        report["outcome"]["exit_status"] == 0,
        report["stderr"],
        report.get("unsupported", []),
    )


def solution_sources(task: Path) -> Iterable[tuple[str, bytes]]:
    solution = task / "solution"
    if not solution.is_dir():
        return
    for path in sorted(solution.rglob("*.py")):
        yield str(path.relative_to(solution)), path.read_bytes()
    for path in sorted(solution.rglob("*.sh")):
        yield from python_heredocs(str(path.relative_to(solution)), path.read_bytes())


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    runner = args.runner.resolve()
    if not runner.is_file():
        raise SystemExit(f"runner does not exist: {runner}")
    tasks = sorted(path.parent for path in source.glob("*/task.toml"))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit(f"no task.toml files found below {source}")

    solution_results: list[tuple[str, str, str]] = []
    verifier_results: list[tuple[str, str]] = []
    feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix="shellsim-tasktrove-") as temporary_name:
        temporary = Path(temporary_name)
        for task in tasks:
            for ordinal, (label, data) in enumerate(solution_sources(task)):
                probe = temporary / task.name / str(ordinal)
                probe.mkdir(parents=True)
                path = probe / "probe.py"
                path.write_bytes(data)
                passed, diagnostic, unsupported = invoke(runner, [str(path)], args.timeout)
                category = "pass" if passed else classify(diagnostic, unsupported, data)
                solution_results.append((task.name, label, category))
                inspected = inspect_python(label, data)
                for imported in inspected.imports:
                    feature_counts[category][f"import:{imported.split('.')[0]}"] += 1
                for node, count in inspected.node_types.items():
                    feature_counts[category][f"ast:{node}"] += count

            tests = task / "tests"
            test_files = sorted(tests.rglob("test_*.py")) if tests.is_dir() else []
            if test_files:
                passed, diagnostic, unsupported = invoke(
                    runner,
                    ["--root", str(task), "--pytest", str(tests)],
                    args.timeout,
                )
                test_source = b"\n".join(path.read_bytes() for path in test_files)
                category = "pass" if passed else classify(diagnostic, unsupported, test_source)
                verifier_results.append((task.name, category))

    solution_categories = Counter(category for _, _, category in solution_results)
    verifier_categories = Counter(category for _, category in verifier_results)
    task_solutions: dict[str, list[str]] = defaultdict(list)
    for task, _label, category in solution_results:
        task_solutions[task].append(category)
    result = {
        "schema_version": 1,
        "method": "first shellsim blocker when loading unchanged sources",
        "source": str(source),
        "tasks": len(tasks),
        "solution_python_sources": len(solution_results),
        "solution_source_passes": solution_categories["pass"],
        "solution_tasks_with_sources": len(task_solutions),
        "solution_tasks_all_sources_pass": sum(
            all(value == "pass" for value in values) for values in task_solutions.values()
        ),
        "solution_failure_categories": dict(
            sorted((key, value) for key, value in solution_categories.items() if key != "pass")
        ),
        "verifier_tasks": len(verifier_results),
        "verifier_task_passes": verifier_categories["pass"],
        "verifier_failure_categories": dict(
            sorted((key, value) for key, value in verifier_categories.items() if key != "pass")
        ),
        "solution_failures": [
            {"task": task, "source": label, "category": category}
            for task, label, category in solution_results
            if category != "pass"
        ],
        "failure_feature_counts": {
            category: dict(counts.most_common()) for category, counts in sorted(feature_counts.items())
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
