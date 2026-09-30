"""A Python host registers tools while a package-owned Python guest calls them over HTTP."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path

import pytest
import shellsim


GUEST_SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "host_tools"
ENDPOINT = "http://host.shellsim/tools"


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
    for name in ("host_tools.py", "grader.py"):
        container.write_file(f"/work/{name}", (GUEST_SOURCE / name).read_bytes())

    result = container.run("python3.14 /work/grader.py")
    assert (result.returncode, result.stdout, result.stderr) == (0, b"1.0\n", b"")
    assert submissions == [1.0]
    assert calls == ["conversation.list", "workspace.read_file", "grade.submit"]

    message = json.dumps({"tool": "conversation.list", "arguments": {}})
    curl = (
        "curl -s -X POST -H 'Content-Type: application/json' -d "
        f"{shlex.quote(message)} {ENDPOINT}"
    )
    result = container.run(curl)
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

    missing = (
        "from host_tools import ToolClient\n"
        "try:\n"
        "    ToolClient().call(\"missing.tool\", {})\n"
        "except RuntimeError as error:\n"
        "    print(str(error))\n"
    )
    result = container.run(f"python3.14 -c {shlex.quote(missing)}")
    assert (result.returncode, result.stdout, result.stderr) == (0, b"unknown tool\n", b"")


def test_container_validates_tool_registration_and_hides_unexpected_errors() -> None:
    with pytest.raises(TypeError, match="tools must be a mapping"):
        shellsim.Container(tools=["not a mapping"])
    with pytest.raises(TypeError, match="callable handlers"):
        shellsim.Container(tools={"broken": object()})

    def fails(arguments: dict[str, object]) -> dict[str, object]:
        raise ValueError("sensitive host diagnostic")

    container = shellsim.Container(tools={"fails": fails})
    source = (
        "import json\n"
        "from urllib.request import Request, urlopen\n"
        f"request = Request({ENDPOINT!r}, data=json.dumps({{'tool':'fails','arguments':{{}}}}).encode())\n"
        "with urlopen(request) as response:\n"
        "    print(response.read().decode())\n"
    )
    result = container.run(f"python3.14 -c {shlex.quote(source)}")
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"error": "tool failed"}
    assert b"sensitive" not in result.stdout + result.stderr

    def denied(arguments: dict[str, object]) -> dict[str, object]:
        raise shellsim.ToolError("not permitted")

    container = shellsim.Container(tools={"fails": denied})
    result = container.run(f"python3.14 -c {shlex.quote(source)}")
    assert json.loads(result.stdout) == {"error": "not permitted"}


def test_host_tool_registry_is_scoped_to_each_container() -> None:
    first = shellsim.Container(tools={"identity": lambda arguments: {"name": "first"}})
    second = shellsim.Container(tools={"identity": lambda arguments: {"name": "second"}})
    source = (
        "import json\n"
        "from urllib.request import Request, urlopen\n"
        f"request = Request({ENDPOINT!r}, data=json.dumps({{'tool':'identity','arguments':{{}}}}).encode())\n"
        "with urlopen(request) as response:\n"
        "    print(json.loads(response.read().decode())['result']['name'])\n"
    )
    command = f"python3.14 -c {shlex.quote(source)}"

    assert first.run(command).stdout == b"first\n"
    assert second.run(command).stdout == b"second\n"
