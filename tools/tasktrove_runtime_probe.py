#!/usr/bin/env python3
"""Sample TaskTrove golden solutions and verifiers in shellsim.

This is a prioritization probe, not a TaskTrove runner. It mounts each task's environment source
at the Dockerfile's last WORKDIR, runs the golden solution, then runs the verifier in the same
shellsim environment. It intentionally does not build Docker images or install dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

WORKDIR = re.compile(r"^\s*WORKDIR\s+(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Action:
    """Small stable subset of one shellsim result."""

    phase: str
    returncode: int
    stop_reason: Optional[str]
    unsupported: tuple[str, ...]
    unsupported_commands: tuple[str, ...]
    commands: tuple[str, ...]
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted TaskTrove task-directory root")
    parser.add_argument("--task", action="append", default=[], help="sample only this task (repeatable)")
    parser.add_argument("--limit", type=int, help="sample only the first N selected tasks")
    parser.add_argument("--quiet", action="store_true", help="do not print task progress")
    return parser.parse_args()


def task_workdir(task: Path) -> str:
    """Use the last literal Docker WORKDIR, falling back to /work."""
    dockerfile = task / "environment" / "Dockerfile"
    matches = WORKDIR.findall(dockerfile.read_text(errors="replace")) if dockerfile.is_file() else []
    value = matches[-1] if matches else "/work"
    return value if value.startswith("/") and "$" not in value else "/work"


def observe(phase: str, result: Any) -> Action:
    stderr = result.stderr_text
    if len(stderr) > 2_000:
        stderr = stderr[:2_000] + "\n[truncated]"
    return Action(
        phase=phase,
        returncode=result.returncode,
        stop_reason=result.stop_reason,
        unsupported=tuple(result.unsupported),
        unsupported_commands=tuple(result.unsupported_commands),
        commands=tuple(result.commands),
        stderr=stderr,
    )


def run(environment: Any, actions: list[Action], phase: str, source: str) -> Any:
    result = environment.run(source)
    actions.append(observe(phase, result))
    return result


def verifier_payload(tests: Path) -> Optional[str]:
    python_tests = sorted(tests.rglob("test_*.py"))
    if python_tests:
        paths = [f"/tests/{path.relative_to(tests).as_posix()}" for path in python_tests]
        return "cd /tests && pytest " + " ".join(shlex.quote(path) for path in paths)
    if (tests / "ref_eval.py").is_file():
        return "cd /tests && python /tests/ref_eval.py"
    return None


def phase_outcome(result: Any) -> str:
    if result is None:
        return "not_run"
    if result.stop_reason:
        return "resource_exhausted"
    if result.unsupported or result.unsupported_commands:
        return "explicit_boundary"
    return "passed" if result.returncode == 0 else "failed"


def reward(environment: Any) -> Optional[str]:
    for path in ("/logs/verifier/reward.txt", "/log/reward.txt"):
        try:
            return environment.read_file(path).decode(errors="replace").strip()
        except Exception:  # absent VFS paths use the public adapter error
            pass
    return None


def replay(task: Path, environment_type: Any, limits_type: Any) -> dict[str, Any]:
    environment = environment_type(limits_type(cpu=100_000_000, memory=256 << 20, disk=256 << 20, output=8 << 20))
    actions: list[Action] = []
    errors: list[str] = []
    workdir = task_workdir(task)

    try:
        environment.mkdir(workdir, parents=True)
        environment.mount(task / "environment", workdir)
        run(environment, actions, "setup", f"cd {shlex.quote(workdir)}")
    except Exception as error:
        errors.append(f"environment: {error}")

    solution_result = None
    solution = task / "solution"
    if not environment.terminated and (solution / "solve.sh").is_file():
        try:
            environment.mount(solution, "/solution")
            solution_result = run(
                environment,
                actions,
                "solution",
                f"cd {shlex.quote(workdir)} && bash /solution/solve.sh",
            )
        except Exception as error:
            errors.append(f"solution: {error}")

    wrapper_result = None
    verifier_result = None
    verifier_source = None
    tests = task / "tests"
    if not environment.terminated and tests.is_dir():
        try:
            environment.mount(tests, "/tests")
            wrapper = tests / "test.sh"
            if wrapper.is_file():
                wrapper_result = run(environment, actions, "verifier_wrapper", "cd /tests && bash /tests/test.sh")
            payload = verifier_payload(tests)
            wrapper_clean = (
                wrapper_result is not None
                and wrapper_result.returncode == 0
                and not wrapper_result.stop_reason
                and not wrapper_result.unsupported
                and not wrapper_result.unsupported_commands
            )
            if wrapper_clean or payload is None:
                verifier_result = wrapper_result
                verifier_source = "test.sh" if wrapper_result is not None else None
            elif not environment.terminated:
                run(environment, actions, "probe", "rm -f /logs/verifier/reward.txt /log/reward.txt")
                verifier_result = run(environment, actions, "verifier", payload)
                verifier_source = "normalized_python"
        except Exception as error:
            errors.append(f"verifier: {error}")

    return {
        "task": task.name,
        "workdir": workdir,
        "solution_outcome": phase_outcome(solution_result),
        "verifier_wrapper_outcome": phase_outcome(wrapper_result),
        "verifier_outcome": phase_outcome(verifier_result),
        "verifier_source": verifier_source,
        "reward": reward(environment),
        "errors": errors,
        "actions": [asdict(action) for action in actions],
    }


def gap_name(value: str) -> str:
    module = re.search(r"no module named [\"']([^\"']+)", value)
    if module:
        return f"python-module:{module.group(1).split('.')[0]}"
    if value.startswith("python:"):
        return re.split(r" (?:at line|in /)", value, maxsplit=1)[0][:160]
    if value.startswith("pip:"):
        return re.split(r" at line", value, maxsplit=1)[0][:160]
    return value[:160]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    phases = ("solution_outcome", "verifier_wrapper_outcome", "verifier_outcome")
    outcomes = {phase: dict(sorted(Counter(result[phase] for result in results).items())) for phase in phases}
    gap_tasks: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for result in results:
        for action in result["actions"]:
            gaps = {gap_name(value) for value in [*action["unsupported"], *action["unsupported_commands"]]}
            for gap in gaps:
                gap_tasks[action["phase"]][gap].add(result["task"])
    gaps = {
        phase: dict(sorted(((gap, len(tasks)) for gap, tasks in values.items()), key=lambda item: (-item[1], item[0])))
        for phase, values in sorted(gap_tasks.items())
    }
    return {
        "outcomes": outcomes,
        "gap_task_counts": gaps,
        "tasks_with_errors": sum(bool(result["errors"]) for result in results),
    }


def main() -> int:
    options = parse_args()
    source = options.source.resolve()
    tasks = sorted(path.parent for path in source.glob("*/task.toml"))
    if options.task:
        selected = set(options.task)
        tasks = [task for task in tasks if task.name in selected]
    if options.limit is not None:
        tasks = tasks[: options.limit]
    if not tasks:
        raise SystemExit(f"no selected tasks found below {source}")

    try:
        from shellsim import Environment, Limits
    except ImportError as error:
        raise SystemExit("install the local shellsim Python package before running this probe") from error

    results = []
    for index, task in enumerate(tasks, 1):
        if not options.quiet:
            print(f"[{index}/{len(tasks)}] {task.name}", file=sys.stderr)
        results.append(replay(task, Environment, Limits))
    print(
        json.dumps(
            {
                "schema_version": 3,
                "method": "approximate task mount; golden solution; verifier wrapper with Python fallback",
                "source": str(source),
                "tasks": len(results),
                "summary": summarize(results),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
