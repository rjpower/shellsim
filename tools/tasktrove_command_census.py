#!/usr/bin/env python3
"""Approximate shell command positions in an extracted TaskTrove-style corpus.

This is a static prioritization tool, not a shell parser. It tokenizes shell and Bash files,
classifies apparent command names against shellsim's Rust registry, and emits deterministic JSON.
Dynamic expansions, function calls, generated scripts, and shell syntax can remain ``missing``;
inspect those rows before treating them as absent binaries.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path

CONTROL = {"\n", ";", "&&", "||", "|", "&", "(", ")", "{", "}"}
RESERVED = {
    "!",
    "case",
    "do",
    "done",
    "elif",
    "else",
    "esac",
    "fi",
    "for",
    "function",
    "if",
    "in",
    "select",
    "then",
    "time",
    "until",
    "while",
}
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\+)?=")
WORD = re.compile(r"^[A-Za-z0-9_./+:-]+$")
HEREDOC = re.compile(r"<<-?\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted task-directory root")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="shellsim checkout used for classification (default: current directory)",
    )
    return parser.parse_args()


def shell_units(text: str) -> list[str]:
    """Return the outer shell plus nested heredocs with a shell shebang."""
    outer: list[str] = []
    nested: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        outer.append(line)
        match = HEREDOC.search(line)
        if match:
            delimiter = next(group for group in match.groups() if group is not None)
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != delimiter:
                body.append(lines[index])
                index += 1
            first = next((item.strip() for item in body if item.strip()), "")
            if re.match(r"^#!.*\b(?:ba|z|da)?sh\b", first):
                nested.append("\n".join(body))
        index += 1
    units = ["\n".join(outer)]
    for body in nested:
        units.extend(shell_units(body))
    return units


def shell_commands(text: str) -> list[str]:
    """Collect conservative executable-position words from one shell source."""
    lexer = shlex.shlex(text.replace("\\\n", " "), posix=True, punctuation_chars="|&;(){}<>\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        tokens = list(lexer)
    except ValueError:
        return []

    commands: list[str] = []
    command_position = True
    skip_redirection_target = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in CONTROL or set(token) <= set("|&;(){}\n"):
            command_position = True
            index += 1
            continue
        if token.startswith((">", "<")):
            skip_redirection_target = token in {">", ">>", "<", "<<", "<<-", "<>"}
            index += 1
            continue
        if skip_redirection_target:
            skip_redirection_target = False
            index += 1
            continue
        if token in {"then", "do", "else", "elif"}:
            command_position = True
            index += 1
            continue
        if token in RESERVED:
            command_position = token in {"if", "elif", "while", "until", "time", "!"}
            index += 1
            continue
        if command_position and ASSIGNMENT.match(token):
            index += 1
            continue
        if command_position:
            if index + 2 < len(tokens) and tokens[index + 1 : index + 3] == ["(", ")"]:
                command_position = False
                index += 3
                continue
            if WORD.match(token) and not token.startswith(("$", "-")):
                commands.append(token.rsplit("/", 1)[-1])
            command_position = False
        index += 1
    return commands


def command_registry(repository: Path) -> dict[str, str]:
    """Read literal command registrations without compiling or executing shellsim."""
    result: dict[str, str] = {}
    for path in sorted((repository / "src" / "commands").glob("*.rs")):
        text = path.read_text()
        for match in re.finditer(r"reg_unsupported\s*\([^;]*?&\[([^]]*)\]", text, re.S):
            for name in re.findall(r'"([^"]+)"', match.group(1)):
                result[name] = "unsupported"
        pattern = (
            r"reg(?:_resumable|_buffered_resumable|_costed)?\s*\([^;]*?"
            r"&\[([^]]*)\][^;]*?Trust::(Real|Partial|Unsupported)"
        )
        for match in re.finditer(pattern, text, re.S):
            for name in re.findall(r'"([^"]+)"', match.group(1)):
                result[name] = match.group(2).lower()
    return result


def source_kind(source: Path, path: Path) -> str:
    parts = path.relative_to(source).parts
    if "solution" in parts:
        return "solution"
    if "tests" in parts:
        return "tests"
    if "environment" in parts:
        return "environment"
    return "other"


def census(source: Path, repository: Path) -> dict[str, object]:
    registry = command_registry(repository)
    paths = sorted(set(source.rglob("*.sh")) | set(source.rglob("*.bash")))
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    files: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in paths:
        bucket = source_kind(source, path)
        for unit in shell_units(path.read_text(errors="replace")):
            for command in shell_commands(unit):
                counts[bucket][command] += 1
                files[bucket][command].add(str(path.relative_to(source)))

    buckets: dict[str, object] = {}
    for bucket in ("solution", "tests", "environment", "other"):
        statuses: Counter[str] = Counter()
        commands = []
        for command, count in counts[bucket].most_common():
            status = registry.get(command, "missing")
            statuses[status] += count
            commands.append(
                {
                    "command": command,
                    "invocations": count,
                    "files": len(files[bucket][command]),
                    "status": status,
                }
            )
        buckets[bucket] = {
            "shell_files": sum(source_kind(source, path) == bucket for path in paths),
            "invocations": sum(counts[bucket].values()),
            "unique_commands": len(counts[bucket]),
            "status_counts": dict(sorted(statuses.items())),
            "commands": commands,
        }
    return {"schema_version": 1, "method": "static executable-position approximation", "buckets": buckets}


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    repository = args.repository.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")
    if not (repository / "src" / "commands").is_dir():
        raise SystemExit(f"not a shellsim repository: {repository}")
    print(json.dumps(census(source, repository), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
