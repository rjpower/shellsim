#!/usr/bin/env python3
"""Approximate shell command surfaces in an extracted TaskTrove-style corpus.

This is a static prioritization tool, not a shell parser. It tokenizes shell and Bash files,
classifies apparent command names against shellsim's Rust registry, groups normalized argv shapes,
and inventories options and embedded-language features. Dynamic expansions, function calls,
generated scripts, and shell syntax can remain ``missing`` or ``dynamic``; inspect those rows
before treating them as absent binaries or unsupported interfaces.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
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
DYNAMIC = re.compile(r"(?:\$[{(A-Za-z_]|`)")
DYNAMIC_PROGRAM = re.compile(r"(?:\$\(|\$\{|`)")
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass(frozen=True)
class Invocation:
    """One statically visible executable-position word and its shell-token arguments."""

    argv: tuple[str, ...]


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


def shell_invocations(text: str) -> list[Invocation]:
    """Collect conservative command invocations from one shell source.

    Redirection targets are excluded from argv. Shell expansion is never evaluated.
    """
    lexer = shlex.shlex(text.replace("\\\n", " "), posix=True, punctuation_chars="|&;(){}<>\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        tokens = list(lexer)
    except ValueError:
        return []

    invocations: list[Invocation] = []
    command_position = True
    skip_redirection_target = False
    current: list[str] = []

    def finish() -> None:
        if current:
            invocations.append(Invocation(tuple(current)))
            current.clear()

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in CONTROL or set(token) <= set("|&;(){}\n"):
            finish()
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
            finish()
            command_position = True
            index += 1
            continue
        if token in RESERVED:
            finish()
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
                current.append(token.rsplit("/", 1)[-1])
            elif DYNAMIC.search(token):
                current.extend(("<dynamic>", token))
            command_position = False
        elif current:
            current.append(token)
        index += 1
    finish()
    return invocations


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


def dynamic_command_kind(argument: str) -> str:
    """Describe why an executable-position token cannot be resolved statically."""
    if "$(" in argument:
        return "command-substitution"
    if "`" in argument:
        return "backtick-substitution"
    return "variable"


# Options whose following token is data rather than another option. This table is used only to
# normalize equivalent corpus calls; it is not a declaration that shellsim supports the option.
VALUE_OPTIONS = {
    "awk": {"-F", "-v", "-f", "--field-separator", "--assign", "--file"},
    "grep": {
        "-e",
        "-f",
        "-m",
        "-A",
        "-B",
        "-C",
        "--regexp",
        "--file",
        "--max-count",
        "--include",
        "--exclude",
        "--exclude-dir",
        "--before-context",
        "--after-context",
        "--context",
    },
    "find": {"-name", "-path", "-wholename", "-type", "-mindepth", "-maxdepth"},
    "jq": {"-f", "--from-file", "--arg", "--argjson", "--slurpfile", "--rawfile", "--indent"},
    "python": {"-c", "-m", "-W", "-X"},
    "python3": {"-c", "-m", "-W", "-X"},
    "python3.14": {"-c", "-m", "-W", "-X"},
    "sed": {"-e", "-f", "--expression", "--file"},
}
OPTIONAL_ATTACHED_OPTIONS = {"sed": {"-i", "--in-place"}}
SHORT_CLUSTER_COMMANDS = {"awk", "grep", "jq", "python", "python3", "python3.14", "sed"}
VALUE_OPTION_ARITY = {("jq", "--arg"): 2, ("jq", "--argjson"): 2, ("jq", "--slurpfile"): 2, ("jq", "--rawfile"): 2}


@dataclass(frozen=True)
class Surface:
    """Normalized option and embedded-language surface of an invocation."""

    shape: str
    options: tuple[str, ...]
    features: tuple[str, ...]
    dynamic: bool


@dataclass(frozen=True)
class CompatibilityProfile:
    """One command's intentionally reviewed static compatibility boundary."""

    options: frozenset[str]
    features: frozenset[str]
    functions: frozenset[str]
    operands: frozenset[str]


def compatibility_profiles() -> dict[str, CompatibilityProfile]:
    """Load independently maintained command profiles and reject duplicate registrations."""
    directory = Path(__file__).with_name("tasktrove_command_profiles")
    profiles: dict[str, CompatibilityProfile] = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1 or not isinstance(data.get("command"), str):
            raise ValueError(f"invalid compatibility profile: {path}")
        command = data["command"]
        if command in profiles:
            raise ValueError(f"duplicate compatibility profile for {command!r}")
        profiles[command] = CompatibilityProfile(
            options=frozenset(data.get("options", [])),
            features=frozenset(data.get("features", [])),
            functions=frozenset(data.get("functions", [])),
            operands=frozenset(data.get("operands", [])),
        )
    if not profiles:
        raise ValueError(f"no compatibility profiles found in {directory}")
    return profiles


def split_short_options(command: str, argument: str) -> list[tuple[str, str | None]]:
    """Split a short-option cluster, retaining attached values for known value options."""
    if command not in SHORT_CLUSTER_COMMANDS:
        return [(argument, None)]
    values = VALUE_OPTIONS.get(command, set())
    optional = OPTIONAL_ATTACHED_OPTIONS.get(command, set())
    result: list[tuple[str, str | None]] = []
    characters = argument[1:]
    index = 0
    while index < len(characters):
        option = f"-{characters[index]}"
        remainder = characters[index + 1 :]
        if option in values:
            result.append((option, remainder or None))
            return result
        if option in optional:
            result.append((option, remainder if remainder else None))
            return result
        result.append((option, None))
        index += 1
    return result


def option_surface(command: str, arguments: tuple[str, ...]) -> tuple[list[str], list[str], list[str]]:
    """Return normalized option shapes, option names, and positional tokens."""
    value_options = VALUE_OPTIONS.get(command, set())
    optional_options = OPTIONAL_ATTACHED_OPTIONS.get(command, set())
    shapes: list[str] = []
    options: list[str] = []
    positionals: list[str] = []
    options_enabled = True
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if options_enabled and argument == "--":
            shapes.append("--")
            options_enabled = False
        elif options_enabled and argument.startswith("--") and len(argument) > 2:
            option, separator, attached = argument.partition("=")
            options.append(option)
            if separator:
                shapes.append(f"{option}=<value>")
            elif option in value_options:
                shapes.append(f"{option}=<value>")
                index += min(VALUE_OPTION_ARITY.get((command, option), 1), len(arguments) - index - 1)
            elif option in optional_options:
                shapes.append(option)
            else:
                shapes.append(option)
        elif options_enabled and argument.startswith("-") and argument != "-" and not NUMBER.match(argument):
            short = split_short_options(command, argument)
            for option, attached in short:
                options.append(option)
                if option in value_options:
                    shapes.append(f"{option}=<value>")
                    if attached is None and index + 1 < len(arguments):
                        index += min(
                            VALUE_OPTION_ARITY.get((command, option), 1),
                            len(arguments) - index - 1,
                        )
                    if command in {"python", "python3", "python3.14"} and option in {"-c", "-m"}:
                        options_enabled = False
                    break
                if option in optional_options and attached is not None:
                    shapes.append(f"{option}=<value>")
                    break
                shapes.append(option)
        else:
            positionals.append(argument)
        index += 1
    return shapes, options, positionals


def embedded_features(command: str, positionals: list[str], arguments: tuple[str, ...]) -> set[str]:
    """Identify coarse syntax families that options alone cannot describe."""
    if command == "awk":
        program = positionals[0] if positionals else ""
        rules = {
            "conditionals": r"\bif\s*\(",
            "loops": r"\b(?:for|while|do)\b",
            "functions": r"\bfunction\s+[A-Za-z_]",
            "arrays": r"\[[^]]+\]",
            "regex": r"(?:~|!~|/[^/]+/)",
            "getline": r"\bgetline\b",
            "redirection": r"(?:^|[^<>])(?:>>?|\|)\s*[\"$A-Za-z_]",
            "begin-end": r"\b(?:BEGIN|END)\b",
            "next-exit": r"\b(?:next|exit)\b",
        }
    elif command == "sed":
        scripts = []
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if argument in {"-e", "--expression"} and index + 1 < len(arguments):
                scripts.append(arguments[index + 1])
                index += 2
                continue
            if argument.startswith("--expression="):
                scripts.append(argument.partition("=")[2])
            index += 1
        if not scripts and positionals:
            scripts.append(positionals[0])
        program = ";".join(scripts)
        rules = {
            "substitution": r"(?:^|[;{}])\s*s([^A-Za-z0-9\\\s]).*?\1",
            "addresses": r"(?:^|[;{}])\s*(?:\d+|\$|/[^/]+/)",
            "ranges": r"(?:\d+|\$|/[^/]+/)\s*,\s*(?:\d+|\$|/[^/]+/)",
            "branching": r"(?:^|[;{}])\s*(?::|[bt])",
            "hold-space": r"(?:^|[;{}])\s*[hHgGx](?:\s|;|$)",
            "insert-append-change": r"(?:^|[;{}])\s*[iac](?:\\|\s)",
            "delete-print": r"(?:^|[;{}])\s*[dpP](?:\s|;|$)",
            "transliterate": r"(?:^|[;{}])\s*y([^A-Za-z0-9\\\s]).*?\1",
        }
    elif command == "jq":
        program = positionals[0] if positionals else ""
        rules = {
            "pipes": r"\|",
            "iteration": r"\.\[\]",
            "selection": r"\bselect\s*\(",
            "construction": r"[\[{]",
            "conditionals": r"\bif\b.*\bthen\b",
            "variables": r"\$[A-Za-z_]",
            "arithmetic": r"(?:^|\s)[+*/%-](?:\s|\d|\.)",
            "comparison": r"(?:==|!=|<=|>=|<|>)",
            "sorting": r"\bsort(?:_by)?\b",
            "map-reduce": r"\b(?:map|reduce|foreach)\b",
            "definitions": r"\bdef\s+[A-Za-z_]",
            "assignment": r"(?:\|=|\+=|-=|\*=|/=|(?<![=!<>])=(?!=))",
        }
    elif command in {"python", "python3", "python3.14"}:
        features = set()
        source = ""
        if "-c" in arguments:
            features.add("command-source")
            index = arguments.index("-c")
            if index + 1 < len(arguments):
                source = arguments[index + 1]
        elif "-m" in arguments:
            features.add("module-entrypoint")
        elif positionals:
            features.add("script")
        else:
            features.add("stdin-source")
        for module in re.findall(r"(?:^|[;\n])\s*(?:from|import)\s+([A-Za-z_][\w.]*)", source):
            features.add(f"import:{module.split('.', 1)[0]}")
        return features
    else:
        return set()
    features = {feature for feature, pattern in rules.items() if re.search(pattern, program, re.S)}
    if command == "jq":
        keywords = {"and", "else", "end", "if", "or", "select", "then"}
        for function in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", program):
            if function not in keywords:
                features.add(f"call:{function}")
    elif command == "awk":
        definitions = set(re.findall(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)", program))
        keywords = {"for", "if", "print", "printf", "while"}
        for function in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", program):
            if function not in keywords | definitions:
                features.add(f"call:{function}")
    return features


def normalize_positionals(command: str, positionals: list[str], options: list[str]) -> list[str]:
    """Replace corpus-specific operands while retaining their semantic role."""
    roles: list[str] = []
    for index, value in enumerate(positionals):
        if command == "awk" and index == 0:
            roles.append("<program>")
        elif command == "sed" and index == 0:
            roles.append("<script>")
        elif command == "jq" and index == 0:
            roles.append("<filter>")
        elif command == "grep" and index == 0 and not ({"-e", "--regexp"} & set(options)):
            roles.append("<pattern>")
        elif command in {"dd"} and "=" in value:
            roles.append(f"{value.partition('=')[0]}=<value>")
        elif NUMBER.match(value):
            roles.append("<number>")
        elif "/" in value or value.startswith("."):
            roles.append("<path>")
        else:
            roles.append("<arg>")
    return roles


def invocation_surface(invocation: Invocation) -> Surface:
    command, *arguments = invocation.argv
    option_shapes, options, positionals = option_surface(command, tuple(arguments))
    features = embedded_features(command, positionals, tuple(arguments))
    positional_shapes = normalize_positionals(command, positionals, options)
    shape = " ".join([command, *option_shapes, *positional_shapes])
    program = positionals[0] if positionals else ""
    dynamic_surface = command in {"awk", "grep", "jq", "sed"} and bool(
        DYNAMIC_PROGRAM.search(program) or re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", program)
    )
    return Surface(
        shape=shape,
        options=tuple(sorted(set(options))),
        features=tuple(sorted(features)),
        dynamic=dynamic_surface,
    )


def assess_surface(command: str, surface: Surface, profiles: dict[str, CompatibilityProfile]) -> str:
    """Compare a normalized form with a narrow, implementation-backed surface profile."""
    if command in {"python", "python3", "python3.14"}:
        return "unclassified"
    profile = profiles.get(command)
    if profile is None:
        return "unclassified"
    if any(option not in profile.options for option in surface.options):
        return "explicit_boundary"
    if surface.dynamic:
        return "dynamic"
    if command == "dd":
        operands = set(re.findall(r"\b([a-z]+)=<value>", surface.shape))
        if not operands <= profile.operands:
            return "explicit_boundary"
    for feature in surface.features:
        if feature.startswith("call:"):
            function = feature.removeprefix("call:")
            if function not in profile.functions:
                return "explicit_boundary"
        elif feature not in profile.features:
            return "explicit_boundary"
    return "supported_surface"


def census(source: Path, repository: Path) -> dict[str, object]:
    registry = command_registry(repository)
    profiles = compatibility_profiles()
    paths = sorted(set(source.rglob("*.sh")) | set(source.rglob("*.bash")))
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    files: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    tasks: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    surfaces: dict[str, dict[str, Counter[Surface]]] = defaultdict(lambda: defaultdict(Counter))
    surface_files: dict[str, dict[str, dict[Surface, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    surface_tasks: dict[str, dict[str, dict[Surface, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    dynamic_commands: dict[str, Counter[str]] = defaultdict(Counter)
    dynamic_command_files: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    dynamic_command_tasks: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in paths:
        bucket = source_kind(source, path)
        relative = str(path.relative_to(source))
        task = path.relative_to(source).parts[0]
        for unit in shell_units(path.read_text(errors="replace")):
            for invocation in shell_invocations(unit):
                command = invocation.argv[0]
                if command == "<dynamic>":
                    kind = dynamic_command_kind(invocation.argv[1])
                    dynamic_commands[bucket][kind] += 1
                    dynamic_command_files[bucket][kind].add(relative)
                    dynamic_command_tasks[bucket][kind].add(task)
                    continue
                counts[bucket][command] += 1
                files[bucket][command].add(relative)
                tasks[bucket][command].add(task)
                surface = invocation_surface(invocation)
                surfaces[bucket][command][surface] += 1
                surface_files[bucket][command][surface].add(relative)
                surface_tasks[bucket][command][surface].add(task)

    buckets: dict[str, object] = {}
    for bucket in ("solution", "tests", "environment", "other"):
        statuses: Counter[str] = Counter()
        commands = []
        for command, count in counts[bucket].most_common():
            status = registry.get(command, "missing")
            statuses[status] += count
            row: dict[str, object] = {
                "command": command,
                "invocations": count,
                "files": len(files[bucket][command]),
                "tasks": len(tasks[bucket][command]),
                "status": status,
            }
            if status == "partial":
                option_counts: Counter[str] = Counter()
                feature_counts: Counter[str] = Counter()
                dynamic = 0
                compatibility_counts: Counter[str] = Counter()
                compatibility_tasks: dict[str, set[str]] = defaultdict(set)
                forms = []
                ordered_surfaces = sorted(
                    surfaces[bucket][command].items(),
                    key=lambda item: (-item[1], item[0].shape, item[0].features),
                )
                for surface, invocations in ordered_surfaces:
                    dynamic += invocations if surface.dynamic else 0
                    assessment = assess_surface(command, surface, profiles)
                    compatibility_counts[assessment] += invocations
                    compatibility_tasks[assessment].update(surface_tasks[bucket][command][surface])
                    option_counts.update(dict.fromkeys(surface.options, invocations))
                    feature_counts.update(dict.fromkeys(surface.features, invocations))
                    forms.append(
                        {
                            "shape": surface.shape,
                            "invocations": invocations,
                            "files": len(surface_files[bucket][command][surface]),
                            "tasks": len(surface_tasks[bucket][command][surface]),
                            "dynamic": surface.dynamic,
                            "assessment": assessment,
                            "options": list(surface.options),
                            "features": list(surface.features),
                        }
                    )
                row.update(
                    {
                        "dynamic_invocations": dynamic,
                        "compatibility": [
                            {
                                "assessment": assessment,
                                "invocations": compatibility_counts[assessment],
                                "tasks": len(compatibility_tasks[assessment]),
                            }
                            for assessment in sorted(compatibility_counts)
                        ],
                        "options": [
                            {"option": option, "invocations": option_counts[option]} for option in sorted(option_counts)
                        ],
                        "features": [
                            {"feature": feature, "invocations": feature_counts[feature]}
                            for feature in sorted(feature_counts)
                        ],
                        "forms": forms,
                    }
                )
            commands.append(row)
        buckets[bucket] = {
            "shell_files": sum(source_kind(source, path) == bucket for path in paths),
            "invocations": sum(counts[bucket].values()),
            "unique_commands": len(counts[bucket]),
            "dynamic_command_positions": sum(dynamic_commands[bucket].values()),
            "dynamic_commands": [
                {
                    "kind": kind,
                    "invocations": count,
                    "files": len(dynamic_command_files[bucket][kind]),
                    "tasks": len(dynamic_command_tasks[bucket][kind]),
                }
                for kind, count in sorted(dynamic_commands[bucket].items(), key=lambda item: (-item[1], item[0]))
            ],
            "status_counts": dict(sorted(statuses.items())),
            "commands": commands,
        }
    return {
        "schema_version": 2,
        "method": "static executable-position and normalized argv approximation",
        "buckets": buckets,
    }


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
