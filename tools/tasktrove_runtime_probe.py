#!/usr/bin/env python3
"""Sample TaskTrove golden solutions and verifiers in shellsim.

This is a prioritization probe, not a TaskTrove runner. It mounts each task's environment source
at the Dockerfile's last WORKDIR, runs the golden solution, then runs the verifier in the same
shellsim environment. It intentionally does not build Docker images or install dependencies.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

CHMOD = re.compile(r"(?:^|&&|;)\s*(chmod\s+[^&;]+)")


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
    usage: dict[str, int]


@dataclass(frozen=True)
class DockerCopy:
    """One literal host-build-context copy that the probe can reproduce safely."""

    sources: tuple[str, ...]
    destination: str
    destination_is_dir: bool


@dataclass(frozen=True)
class DockerLayout:
    """Small Dockerfile subset used to place trusted fixture files in the VFS."""

    workdir: str
    copies: tuple[DockerCopy, ...]
    chmod_commands: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted TaskTrove task-directory root")
    parser.add_argument("--task", action="append", default=[], help="sample only this task (repeatable)")
    parser.add_argument("--offset", type=int, default=0, help="skip the first N selected tasks")
    parser.add_argument("--limit", type=int, help="sample only the first N selected tasks")
    parser.add_argument(
        "--memory-mib",
        type=int,
        default=256,
        help="modeled memory limit per replay (default: 256)",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print task progress")
    return parser.parse_args()


def docker_instructions(source: str) -> tuple[str, ...]:
    """Join Dockerfile continuations without interpreting shell or substitutions."""
    instructions: list[str] = []
    parts: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not parts and (not stripped or stripped.startswith("#")):
            continue
        continued = stripped.endswith("\\")
        parts.append(stripped[:-1].rstrip() if continued else stripped)
        if not continued:
            instructions.append(" ".join(parts))
            parts = []
    if parts:
        instructions.append(" ".join(parts))
    return tuple(instructions)


def docker_layout(task: Path) -> DockerLayout:
    """Extract literal WORKDIR, build-context COPY, and chmod declarations."""
    dockerfile = task / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return DockerLayout("/work", (), ())

    workdir = "/work"
    copies: list[DockerCopy] = []
    chmod_commands: list[str] = []
    for instruction in docker_instructions(dockerfile.read_text(errors="replace")):
        operation, separator, arguments = instruction.partition(" ")
        if not separator:
            continue
        operation = operation.upper()
        if operation == "WORKDIR":
            fields = shlex.split(arguments)
            if len(fields) == 1 and fields[0].startswith("/") and "$" not in fields[0]:
                workdir = posixpath.normpath(fields[0])
        elif operation == "COPY":
            fields = shlex.split(arguments)
            if fields and fields[0].startswith("--"):
                continue
            if len(fields) >= 2 and all(
                not any(marker in value for marker in ("$", "*", "?", "[")) for value in fields
            ):
                raw_destination = fields[-1]
                destination_is_dir = len(fields) > 2 or raw_destination.endswith("/") or raw_destination in (".", "..")
                destination = raw_destination
                if not destination.startswith("/"):
                    destination = posixpath.join(workdir, destination)
                copies.append(DockerCopy(tuple(fields[:-1]), posixpath.normpath(destination), destination_is_dir))
        elif operation == "RUN":
            chmod_commands.extend(match.group(1).strip() for match in CHMOD.finditer(arguments))
    return DockerLayout(workdir, tuple(copies), tuple(chmod_commands))


def task_workdir(task: Path) -> str:
    """Use the last literal Docker WORKDIR, falling back to /work."""
    return docker_layout(task).workdir


def apply_docker_layout(environment: Any, task: Path, layout: DockerLayout) -> None:
    """Overlay literal Docker COPY destinations and executable modes in modeled state."""
    build_context = (task / "environment").resolve()
    for copy in layout.copies:
        multiple = len(copy.sources) > 1
        for source_text in copy.sources:
            relative = source_text.removeprefix("./")
            source = (build_context / relative).resolve()
            try:
                source.relative_to(build_context)
            except ValueError as error:
                raise ValueError(f"COPY source escapes build context: {source_text}") from error
            if not source.exists() or source.is_symlink():
                continue
            destination = copy.destination
            if source.is_file() and (multiple or copy.destination_is_dir):
                destination = posixpath.join(destination, source.name)
            if source.is_dir():
                environment.mkdir(destination, parents=True)
                environment.mount(source, destination)
            else:
                environment.mkdir(posixpath.dirname(destination), parents=True)
                mode = stat.S_IMODE(source.stat().st_mode)
                environment.write_file(destination, source.read_bytes(), mode=mode)

    for command in layout.chmod_commands:
        environment.run(f"{command} 2>/dev/null || true")


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
        usage=asdict(result.usage),
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


def replay(task: Path, environment_type: Any, limits_type: Any, memory_mib: int = 256) -> dict[str, Any]:
    environment = environment_type(
        limits_type(
            cpu=100_000_000,
            memory=memory_mib << 20,
            disk=256 << 20,
            output=8 << 20,
        )
    )
    actions: list[Action] = []
    errors: list[str] = []
    layout = docker_layout(task)
    workdir = layout.workdir

    try:
        environment.mkdir(workdir, parents=True)
        environment.mount(task / "environment", workdir)
        apply_docker_layout(environment, task, layout)
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
    if options.offset < 0:
        raise SystemExit("--offset must not be negative")
    if options.memory_mib <= 0:
        raise SystemExit("--memory-mib must be positive")
    tasks = tasks[options.offset :]
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
        results.append(replay(task, Environment, Limits, options.memory_mib))
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
