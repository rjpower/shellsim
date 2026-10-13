"""Host-tool registration, failures and isolation through guest HTTP commands."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path

import pytest
import shellsim


GUEST_SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "host_tools"
ENDPOINT = "http://host.shellsim/tools"


def _call(tool: str) -> str:
    message = json.dumps({"tool": tool, "arguments": {}})
    return f"curl -s -X POST -d {shlex.quote(message)} {ENDPOINT}"


def test_python_host_registers_tools_for_guest_and_shell_http() -> None:
    workspace = {"/work/answer.txt": b"42\n"}
    conversation = ["What is 6 * 7?"]
    submissions: list[float] = []
    calls: list[str] = []

    def read_file(arguments: dict[str, object]) -> dict[str, str]:
        calls.append("workspace.read_file")
        return {"data_base64": base64.b64encode(workspace[arguments["path"]]).decode()}

    def list_conversation(arguments: dict[str, object]) -> dict[str, list[str]]:
        assert arguments == {}
        calls.append("conversation.list")
        return {"messages": conversation}

    def submit_grade(arguments: dict[str, object]) -> dict[str, bool]:
        calls.append("grade.submit")
        score = arguments["score"]
        assert isinstance(score, (int, float)) and 0 <= score <= 1
        submissions.append(score)
        return {"accepted": True}

    container = shellsim.Container(
        tools={
            "workspace.read_file": read_file,
            "conversation.list": list_conversation,
            "grade.submit": submit_grade,
        }
    )
    container.write_file("/work/grader.sh", (GUEST_SOURCE / "grader.sh").read_bytes())

    result = container.run("bash /work/grader.sh")
    assert (result.returncode, result.stdout, result.stderr) == (0, b"1.0\n", b"")
    assert submissions == [1.0]
    assert calls == ["conversation.list", "workspace.read_file", "grade.submit"]

    result = container.run("timeout 5 bash /work/grader.sh")
    assert (result.returncode, result.stdout, result.stderr) == (0, b"1.0\n", b"")
    assert submissions == [1.0, 1.0]

    message = json.dumps({"tool": "conversation.list", "arguments": {}})
    curl = (
        "curl -s -X POST -H 'Content-Type: application/json' -d "
        f"{shlex.quote(message)} {ENDPOINT}"
    )
    result = container.run(curl)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"result": {"messages": conversation}}

    result = container.run(f"timeout 5 {curl}")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"result": {"messages": conversation}}

    wget = (
        "wget -q -O - --method POST --header 'Content-Type: application/json' --body-data "
        f"{shlex.quote(message)} {ENDPOINT}"
    )
    result = container.run(wget)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"result": {"messages": conversation}}
    assert calls[-2:] == ["conversation.list", "conversation.list"]

    result = container.run(_call("missing.tool") + " | jq -r .error")
    assert (result.returncode, result.stdout, result.stderr) == (0, b"unknown tool\n", b"")


def test_container_validates_tool_registration_and_hides_unexpected_errors() -> None:
    with pytest.raises(TypeError, match="tools must be a mapping"):
        shellsim.Container(tools=["not a mapping"])
    with pytest.raises(TypeError, match="callable handlers"):
        shellsim.Container(tools={"broken": object()})

    def fails(arguments: dict[str, object]) -> dict[str, object]:
        raise ValueError("sensitive host diagnostic")

    container = shellsim.Container(tools={"fails": fails})
    result = container.run(_call("fails"))
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"error": "tool failed"}
    assert b"sensitive" not in result.stdout + result.stderr

    def denied(arguments: dict[str, object]) -> dict[str, object]:
        raise shellsim.ToolError("not permitted")

    container = shellsim.Container(tools={"fails": denied})
    result = container.run(_call("fails"))
    assert json.loads(result.stdout) == {"error": "not permitted"}


def test_host_tool_registry_is_scoped_to_each_container() -> None:
    first = shellsim.Container(tools={"identity": lambda arguments: {"name": "first"}})
    second = shellsim.Container(tools={"identity": lambda arguments: {"name": "second"}})
    command = _call("identity") + " | jq -r .result.name"

    assert first.run(command).stdout == b"first\n"
    assert second.run(command).stdout == b"second\n"


def test_oversized_tool_result_is_a_guest_error() -> None:
    container = shellsim.Container(tools={"huge": lambda arguments: {"blob": "x" * (1 << 20)}})
    command = _call("huge") + " | jq -r .error"
    result = container.run(command)
    assert (result.returncode, result.stdout, result.stderr) == (0, b"tool response too large\n", b"")

    def huge_error(arguments: dict[str, object]) -> dict[str, object]:
        raise shellsim.ToolError("x" * (1 << 20))

    container = shellsim.Container(tools={"huge": huge_error})
    result = container.run(command)
    assert (result.returncode, result.stdout, result.stderr) == (0, b"tool response too large\n", b"")


def test_completed_action_does_not_invoke_orphan_tool_calls() -> None:
    calls: list[str] = []

    def record(arguments: dict[str, object]) -> dict[str, bool]:
        calls.append("called")
        return {"ok": True}

    container = shellsim.Container(tools={"record": record})
    result = container.run(f"{_call('record')} & printf started")
    assert (result.returncode, result.stdout) == (0, b"started")
    assert calls == []
    assert container.run("printf next").stdout == b"next"
    assert calls == []

    result = container.run(f"(sleep 1; {_call('record')}) & printf started")
    assert (result.returncode, result.stdout) == (0, b"started")
    assert container.run("sleep 2; printf next").stdout == b"next"
    assert calls == []
