"""Exercise the census through its CLI so output schema and static parsing stay coupled."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_census(tmp_path: Path, source: str) -> dict[str, object]:
    corpus = tmp_path / "corpus"
    solution = corpus / "task-one" / "solution"
    solution.mkdir(parents=True)
    (solution / "solve.sh").write_text(source)

    repository = tmp_path / "repository"
    commands = repository / "src" / "commands"
    commands.mkdir(parents=True)
    (commands / "sample.rs").write_text(
        "\n".join(
            f'reg(commands, &["{command}"], Trust::Partial, cmd_{command});'
            for command in ("awk", "dd", "grep", "jq", "sed")
        )
    )
    script = Path(__file__).parents[1] / "tools" / "tasktrove_command_census.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(corpus), "--repository", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def command(result: dict[str, object], name: str) -> dict[str, object]:
    buckets = result["buckets"]
    assert isinstance(buckets, dict)
    solution = buckets["solution"]
    assert isinstance(solution, dict)
    commands = solution["commands"]
    assert isinstance(commands, list)
    return next(row for row in commands if row["command"] == name)


def test_census_groups_options_features_and_explicit_boundaries(tmp_path: Path) -> None:
    result = run_census(
        tmp_path,
        """#!/bin/sh
grep -rn --include='*.rs' needle src
grep -P '\\d+' file
awk 'BEGIN { if ($1 ~ /x/) print $1 }' data
jq -r '.[] | select(.ok)' data.json
jq 'map(.x)' data.json
jq -f filter.jq data.json
sed -n '1,3p' data
sed 'h' data
dd of="$target" bs=1 count=1
$runner --flag
""",
    )

    assert result["schema_version"] == 2
    solution = result["buckets"]["solution"]
    assert solution["dynamic_command_positions"] == 1
    assert solution["invocations"] == 9

    grep = command(result, "grep")
    assert grep["compatibility"] == [
        {"assessment": "explicit_boundary", "invocations": 1, "tasks": 1},
        {"assessment": "supported_surface", "invocations": 1, "tasks": 1},
    ]
    assert {item["option"] for item in grep["options"]} == {"--include", "-P", "-n", "-r"}

    awk = command(result, "awk")
    assert awk["dynamic_invocations"] == 0
    assert {item["feature"] for item in awk["features"]} >= {
        "begin-end",
        "conditionals",
        "regex",
    }

    jq = command(result, "jq")
    assert {item["assessment"] for item in jq["compatibility"]} == {
        "explicit_boundary",
        "supported_surface",
    }
    assert {item["feature"] for item in jq["features"]} >= {"call:map", "iteration", "selection"}

    sed = command(result, "sed")
    assert {item["assessment"] for item in sed["compatibility"]} == {
        "explicit_boundary",
        "supported_surface",
    }

    dd = command(result, "dd")
    assert dd["dynamic_invocations"] == 0
    assert dd["compatibility"] == [
        {"assessment": "supported_surface", "invocations": 1, "tasks": 1}
    ]


def test_census_consumes_multiword_option_values_before_shaping_operands(tmp_path: Path) -> None:
    result = run_census(
        tmp_path,
        """#!/bin/sh
jq -n --arg name value '{$name}'
""",
    )

    jq = command(result, "jq")
    assert jq["forms"] == [
        {
            "assessment": "supported_surface",
            "dynamic": False,
            "features": ["construction", "variables"],
            "files": 1,
            "invocations": 1,
            "options": ["--arg", "-n"],
            "shape": "jq -n --arg=<value> <filter>",
            "tasks": 1,
        }
    ]
