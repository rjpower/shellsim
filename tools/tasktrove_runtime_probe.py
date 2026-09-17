#!/usr/bin/env python3
"""Replay TaskTrove golden solutions and verifier payloads in shellsim.

The probe models a TaskTrove image as shellsim's built-in userspace plus the Dockerfile's local
``COPY``, ``ENV``, ``WORKDIR``, and non-provisioning ``RUN`` steps. Image provisioning commands are
recorded as prerequisites instead of executed: downloading Debian or Python packages is an image
build concern, while the replay is intended to measure shellsim's task-facing behavior. The
unmodified golden ``solve.sh`` and the verifier's Python payload then run in one persistent
environment, so generated files and runtime command names are observed directly.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

KNOWN_INSTRUCTIONS = {
    "ADD",
    "ARG",
    "CMD",
    "COPY",
    "ENTRYPOINT",
    "ENV",
    "EXPOSE",
    "FROM",
    "HEALTHCHECK",
    "LABEL",
    "RUN",
    "SHELL",
    "STOPSIGNAL",
    "USER",
    "VOLUME",
    "WORKDIR",
}
PROVISIONING = re.compile(
    r"\b(?:apt(?:-get)?|apk|yum|dnf|add-apt-repository|update-alternatives|conda|mamba|npm|yarn|pnpm)\b"
    r"|\b(?:python(?:3(?:\.\d+)?)?\s+-m\s+pip|pip3?|pipx|uv\s+pip|gem|poetry)\s+(?:install|add)\b"
    r"|\b(?:cargo|go)\s+install\b|\buv\s+(?:sync|add)\b"
    r"|\b(?:git\s+clone|curl\s+[^|>]*https?://|wget\s+[^|>]*https?://)\b",
    re.IGNORECASE,
)
HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")


@dataclass(frozen=True)
class DockerInstruction:
    """One logical Dockerfile instruction and its source line."""

    operation: str
    argument: str
    line: int


@dataclass(frozen=True)
class Action:
    """Bounded observation from one shellsim action."""

    phase: str
    label: str
    returncode: int
    stop_reason: Optional[str]
    unsupported: tuple[str, ...]
    unsupported_commands: tuple[str, ...]
    commands: tuple[str, ...]
    stderr: str


@dataclass(frozen=True)
class SetupState:
    """Docker-derived state retained even when one setup instruction fails."""

    workdir: str
    prerequisites: tuple[str, ...]
    errors: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted TaskTrove task-directory root")
    parser.add_argument("--task", action="append", default=[], help="replay only this task (repeatable)")
    parser.add_argument("--limit", type=int, help="replay only the first N selected tasks")
    parser.add_argument("--cpu", type=int, default=100_000_000, help="CPU fuel per task")
    parser.add_argument("--memory", type=int, default=256 * 1024 * 1024, help="modeled memory bytes per task")
    parser.add_argument("--disk", type=int, default=256 * 1024 * 1024, help="VFS bytes per task")
    parser.add_argument("--output", type=int, default=8 * 1024 * 1024, help="output bytes per task")
    parser.add_argument("--quiet", action="store_true", help="do not print task progress to stderr")
    return parser.parse_args()


def docker_instructions(text: str) -> list[DockerInstruction]:
    """Parse logical instructions, including continuations and shell heredocs.

    This is deliberately not a general Dockerfile parser. Unknown top-level text is retained as an
    ``UNKNOWN`` instruction so corpus syntax cannot disappear silently.
    """
    lines = text.splitlines()
    instructions: list[DockerInstruction] = []
    index = 0
    while index < len(lines):
        first_line = index + 1
        raw = lines[index]
        index += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"([A-Za-z]+)\s+(.*)", stripped)
        if not match or match.group(1).upper() not in KNOWN_INSTRUCTIONS:
            instructions.append(DockerInstruction("UNKNOWN", stripped, first_line))
            continue
        operation = match.group(1).upper()
        argument = match.group(2)
        while raw.rstrip().endswith("\\") and index < len(lines):
            argument = argument.rstrip()
            argument = argument[:-1].rstrip() + " " + lines[index].strip()
            raw = lines[index]
            index += 1
        delimiter = HEREDOC.search(argument)
        if delimiter:
            body: list[str] = []
            while index < len(lines):
                line = lines[index]
                index += 1
                body.append(line)
                if line.strip() == delimiter.group(1):
                    break
            argument += "\n" + "\n".join(body)
        instructions.append(DockerInstruction(operation, argument, first_line))
    return instructions


def absolute_path(workdir: str, value: str) -> str:
    """Resolve a Docker destination against its current working directory."""
    return posixpath.normpath(value if value.startswith("/") else posixpath.join(workdir, value))


def docker_environment(argument: str) -> dict[str, str]:
    """Parse Docker's assignment and legacy two-word ``ENV`` forms."""
    words = shlex.split(argument)
    if not words:
        raise ValueError("ENV requires a name and value")
    if "=" not in words[0]:
        if len(words) < 2:
            raise ValueError("ENV requires a value")
        return {words[0]: " ".join(words[1:])}
    result: dict[str, str] = {}
    for word in words:
        name, separator, value = word.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid ENV assignment: {word}")
        result[name] = value
    return result


def expand_docker_environment(value: str, environment: dict[str, str]) -> str:
    """Expand the variable forms used by corpus WORKDIR instructions."""

    def replacement(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return environment.get(name, match.group(0))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replacement, value)


def is_image_provisioning(argument: str) -> bool:
    """Return whether a Docker ``RUN`` needs capabilities outside the replay."""
    return bool(PROVISIONING.search(argument) or argument.lstrip().startswith(("[", "--mount=", "<<")))


def copy_local(environment: Any, context: Path, workdir: str, argument: str) -> Optional[str]:
    """Apply the common local-source subset of Docker ``COPY``."""
    try:
        words = shlex.split(argument)
    except ValueError as error:
        return f"COPY parse error: {error}"
    flags = [word for word in words if word.startswith("--")]
    words = [word for word in words if not word.startswith("--")]
    if any(flag.startswith("--from=") for flag in flags):
        return f"multi-stage COPY prerequisite: {argument}"
    if flags:
        return f"unsupported COPY flag: {' '.join(flags)}"
    if len(words) != 2:
        return f"unsupported COPY form: {argument}"
    source_word, destination_word = words
    source = (context / source_word).resolve()
    try:
        source.relative_to(context.resolve())
    except ValueError:
        return f"COPY source escapes build context: {source_word}"
    if not source.exists():
        return f"COPY source does not exist: {source_word}"
    destination = absolute_path(workdir, destination_word)
    if source.is_dir():
        environment.mkdir(destination, parents=True)
        environment.mount(source, destination)
        return None
    if destination_word.endswith("/") or destination_word in {".", "./"} or destination == workdir:
        destination = posixpath.join(destination, source.name)
    environment.mkdir(posixpath.dirname(destination), parents=True)
    environment.write_file(destination, source.read_bytes(), mode=source.stat().st_mode & 0o7777)
    return None


def action_from_result(phase: str, label: str, result: Any) -> Action:
    """Reduce a public ``RunResult`` to stable, bounded replay evidence."""
    stderr = result.stderr_text
    if len(stderr) > 2_000:
        stderr = stderr[:2_000] + "\n[truncated]"
    return Action(
        phase=phase,
        label=label,
        returncode=result.returncode,
        stop_reason=result.stop_reason,
        unsupported=tuple(result.unsupported),
        unsupported_commands=tuple(result.unsupported_commands),
        commands=tuple(result.commands),
        stderr=stderr,
    )


def run_action(environment: Any, actions: list[Action], phase: str, label: str, source: str) -> Any:
    result = environment.run(source)
    actions.append(action_from_result(phase, label, result))
    return result


def prepare_environment(environment: Any, task: Path, actions: list[Action]) -> SetupState:
    """Materialize local image data and execute deterministic Docker setup in order."""
    context = task / "environment"
    dockerfile = context / "Dockerfile"
    workdir = "/"
    prerequisites: list[str] = []
    errors: list[str] = []
    docker_env: dict[str, str] = {}
    stages = 0
    for instruction in docker_instructions(dockerfile.read_text(errors="replace")):
        if environment.terminated:
            break
        operation, argument = instruction.operation, instruction.argument
        label = f"Dockerfile:{instruction.line}"
        try:
            if operation == "FROM":
                stages += 1
                if stages > 1:
                    prerequisites.append(f"{label}: additional image stage {argument}")
            elif operation == "WORKDIR":
                expanded = expand_docker_environment(argument.strip(), docker_env)
                if "$" in expanded:
                    prerequisites.append(f"{label}: unresolved WORKDIR variable: {argument}")
                else:
                    workdir = absolute_path(workdir, expanded)
                    environment.mkdir(workdir, parents=True)
                    run_action(environment, actions, "setup", label, f"cd {shlex.quote(workdir)}")
            elif operation == "ENV":
                parsed = docker_environment(argument)
                values = {
                    name: expand_docker_environment(value, {**docker_env, **parsed}) for name, value in parsed.items()
                }
                docker_env.update(values)
                exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in values.items())
                run_action(environment, actions, "setup", label, f"export {exports}")
            elif operation == "COPY":
                error = copy_local(environment, context, workdir, argument)
                if error:
                    prerequisites.append(f"{label}: {error}")
            elif operation == "ADD":
                prerequisites.append(f"{label}: ADD is not materialized: {argument}")
            elif operation == "RUN":
                if is_image_provisioning(argument):
                    prerequisites.append(f"{label}: image provisioning: {argument.splitlines()[0]}")
                else:
                    run_action(environment, actions, "setup", label, argument)
            elif operation == "USER" and argument.strip() not in {"root", "0", "0:0"}:
                prerequisites.append(f"{label}: user identity not modeled: {argument}")
            elif operation == "UNKNOWN":
                prerequisites.append(f"{label}: unparsed Dockerfile text: {argument}")
        except Exception as error:
            errors.append(f"{label}: {error}")
    if not environment.terminated:
        try:
            environment.mkdir(workdir, parents=True)
            run_action(environment, actions, "setup", "final WORKDIR", f"cd {shlex.quote(workdir)}")
        except Exception as error:
            errors.append(f"final WORKDIR: {error}")
    return SetupState(workdir, tuple(prerequisites), tuple(errors))


def verifier_command(tests: Path) -> Optional[str]:
    """Select the verifier payload without package-installing harness boilerplate."""
    python_tests = sorted(tests.rglob("test_*.py"))
    if python_tests:
        paths = [f"/tests/{path.relative_to(tests).as_posix()}" for path in python_tests]
        return "cd /tests && pytest " + " ".join(shlex.quote(path) for path in paths)
    reference = tests / "ref_eval.py"
    if reference.is_file():
        return "cd /tests && python /tests/ref_eval.py"
    script = tests / "test.sh"
    if script.is_file():
        return "cd /tests && bash /tests/test.sh"
    return None


def read_reward(environment: Any) -> Optional[str]:
    for path in ("/logs/verifier/reward.txt", "/log/reward.txt"):
        try:
            return environment.read_file(path).decode("utf-8", errors="replace").strip()
        except Exception:  # the public adapter raises SimulationError for absent VFS paths
            pass
    return None


def classify(actions: list[Action], verifier_returncode: Optional[int], reward: Optional[str]) -> str:
    """Classify observed execution without guessing about unseen behavior."""
    for action in actions:
        if action.stop_reason:
            return "resource_exhausted"
    if verifier_returncode is None:
        return "no_verifier"
    boundary = any(
        (action.unsupported or action.unsupported_commands)
        and action.phase not in {"verifier_wrapper", "verifier_harness"}
        for action in actions
    )
    if reward is not None:
        try:
            score = float(reward)
            if score <= 0:
                return "boundary_and_verifier_failed" if boundary else "verifier_failed"
            if score < 1:
                return "partial_reward_with_boundary" if boundary else "partial_reward"
        except ValueError:
            return "invalid_reward"
        return "passed_with_boundary" if boundary else "passed"
    if verifier_returncode == 0:
        return "passed_with_boundary" if boundary else "passed"
    if boundary:
        return "explicit_boundary"
    return "verifier_failed"


def phase_outcome(result: Any) -> str:
    """Describe one phase independently of later verifier evidence."""
    if result is None:
        return "not_run"
    if result.stop_reason:
        return "resource_exhausted"
    if result.unsupported or result.unsupported_commands:
        return "explicit_boundary"
    return "passed" if result.returncode == 0 else "failed"


def replay_task(task: Path, environment_type: Any, limits_type: Any, options: argparse.Namespace) -> dict[str, Any]:
    environment = environment_type(
        limits_type(cpu=options.cpu, memory=options.memory, disk=options.disk, output=options.output)
    )
    actions: list[Action] = []
    harness_errors: list[str] = []
    workdir = "/"
    try:
        setup = prepare_environment(environment, task, actions)
        workdir = setup.workdir
        prerequisites = list(setup.prerequisites)
        harness_errors.extend(setup.errors)
    except Exception as error:
        prerequisites = []
        harness_errors.append(f"environment setup: {error}")

    solution = task / "solution"
    solution_result = None
    if not solution.is_dir():
        harness_errors.append("solution directory is absent")
    elif not environment.terminated:
        try:
            environment.mount(solution, "/solution")
            solve = solution / "solve.sh"
            if solve.is_file():
                source = f"cd {shlex.quote(workdir)} && bash /solution/solve.sh"
                solution_result = run_action(environment, actions, "solution", "solution/solve.sh", source)
            else:
                harness_errors.append("solution/solve.sh is absent")
        except Exception as error:
            harness_errors.append(f"solution mount/run: {error}")

    tests = task / "tests"
    wrapper_result = None
    verifier_result = None
    verifier_source = None
    command = verifier_command(tests) if tests.is_dir() else None
    if command is None:
        harness_errors.append("verifier payload is absent")
    elif not environment.terminated:
        try:
            environment.mount(tests, "/tests")
            wrapper = tests / "test.sh"
            if wrapper.is_file():
                wrapper_result = run_action(
                    environment,
                    actions,
                    "verifier_wrapper",
                    "tests/test.sh",
                    "cd /tests && bash /tests/test.sh",
                )
            wrapper_is_authoritative = (
                wrapper_result is not None
                and wrapper_result.returncode == 0
                and not wrapper_result.stop_reason
                and not wrapper_result.unsupported
                and not wrapper_result.unsupported_commands
            )
            if wrapper_is_authoritative:
                verifier_result = wrapper_result
                verifier_source = "test.sh"
            elif not environment.terminated:
                run_action(
                    environment,
                    actions,
                    "verifier_harness",
                    "clear wrapper reward",
                    "rm -f /logs/verifier/reward.txt /log/reward.txt",
                )
                verifier_result = run_action(environment, actions, "verifier", "normalized payload", command)
                verifier_source = "normalized_payload"
        except Exception as error:
            harness_errors.append(f"verifier mount/run: {error}")

    reward = read_reward(environment)
    category = (
        "harness_error"
        if harness_errors and verifier_result is None
        else classify(actions, None if verifier_result is None else verifier_result.returncode, reward)
    )
    task_boundaries = [
        action
        for action in actions
        if action.phase not in {"verifier_wrapper", "verifier_harness"}
        and (action.unsupported or action.unsupported_commands)
    ]
    wrapper_boundaries = [
        action
        for action in actions
        if action.phase == "verifier_wrapper" and (action.unsupported or action.unsupported_commands)
    ]

    def boundary_record(candidates: list[Action]) -> Optional[dict[str, Any]]:
        if not candidates:
            return None
        action = candidates[0]
        return {
            "phase": action.phase,
            "label": action.label,
            "unsupported": list(action.unsupported),
            "commands": list(action.unsupported_commands),
        }

    first_boundary = boundary_record(task_boundaries)
    first_wrapper_boundary = boundary_record(wrapper_boundaries)
    return {
        "task": task.name,
        "category": category,
        "workdir": workdir,
        "solution_returncode": None if solution_result is None else solution_result.returncode,
        "solution_outcome": phase_outcome(solution_result),
        "verifier_wrapper_outcome": phase_outcome(wrapper_result),
        "verifier_returncode": None if verifier_result is None else verifier_result.returncode,
        "verifier_outcome": phase_outcome(verifier_result),
        "verifier_source": verifier_source,
        "reward": reward,
        "first_boundary": first_boundary,
        "first_wrapper_boundary": first_wrapper_boundary,
        "image_prerequisites": prerequisites,
        "harness_errors": harness_errors,
        "actions": [asdict(action) for action in actions],
    }


def main() -> int:
    options = parse_args()
    source = options.source.resolve()
    tasks = sorted(path.parent for path in source.glob("*/task.toml"))
    if options.task:
        selected = set(options.task)
        tasks = [task for task in tasks if task.name in selected]
        missing = sorted(selected - {task.name for task in tasks})
        if missing:
            raise SystemExit(f"unknown task(s): {', '.join(missing)}")
    if options.limit is not None:
        tasks = tasks[: options.limit]
    if not tasks:
        raise SystemExit(f"no task.toml files found below {source}")

    try:
        from shellsim import Environment, Limits
    except ImportError as error:
        raise SystemExit("install the local shellsim Python package before running this probe") from error

    results = []
    for index, task in enumerate(tasks, 1):
        if not options.quiet:
            print(f"[{index}/{len(tasks)}] {task.name}", file=sys.stderr)
        results.append(replay_task(task, Environment, Limits, options))
    categories = Counter(result["category"] for result in results)
    solution_outcomes = Counter(result["solution_outcome"] for result in results)
    wrapper_outcomes = Counter(result["verifier_wrapper_outcome"] for result in results)
    verifier_outcomes = Counter(result["verifier_outcome"] for result in results)
    verifier_sources = Counter(result["verifier_source"] or "not_run" for result in results)
    output = {
        "schema_version": 2,
        "method": "golden solution, verifier wrapper, and normalized fallback in one shellsim environment",
        "source": str(source),
        "tasks": len(results),
        "categories": dict(sorted(categories.items())),
        "solution_outcomes": dict(sorted(solution_outcomes.items())),
        "verifier_wrapper_outcomes": dict(sorted(wrapper_outcomes.items())),
        "verifier_outcomes": dict(sorted(verifier_outcomes.items())),
        "verifier_sources": dict(sorted(verifier_sources.items())),
        "tasks_with_image_prerequisites": sum(bool(result["image_prerequisites"]) for result in results),
        "results": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
