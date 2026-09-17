"""Tests for the execution-first TaskTrove replay harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "tools" / "tasktrove_runtime_probe.py"
SPEC = importlib.util.spec_from_file_location("tasktrove_runtime_probe", SCRIPT)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def test_docker_instructions_preserve_continuations_and_heredocs() -> None:
    instructions = PROBE.docker_instructions(
        """FROM python:3.13
RUN printf '%s' \\
  value > /tmp/value
RUN cat > /tmp/data <<'EOF'
{"value": 1}
EOF
WORKDIR /app
"""
    )

    assert [(item.operation, item.line) for item in instructions] == [
        ("FROM", 1),
        ("RUN", 2),
        ("RUN", 4),
        ("WORKDIR", 7),
    ]
    assert instructions[1].argument == "printf '%s' value > /tmp/value"
    assert instructions[2].argument.endswith('{"value": 1}\nEOF')


class FakeEnvironment:
    def __init__(self) -> None:
        self.mounts: list[tuple[Path, str]] = []
        self.files: list[tuple[str, bytes, int]] = []

    def mkdir(self, _destination: str, *, parents: bool) -> None:
        assert parents

    def mount(self, source: Path, destination: str) -> None:
        self.mounts.append((source, destination))

    def write_file(self, destination: str, data: bytes, *, mode: int) -> None:
        self.files.append((destination, data, mode))


def test_copy_local_maps_directories_and_files_to_docker_destinations(tmp_path: Path) -> None:
    context = tmp_path / "environment"
    (context / "data").mkdir(parents=True)
    (context / "data" / "input.txt").write_text("payload")
    environment = FakeEnvironment()

    assert PROBE.copy_local(environment, context, "/app", "data /workdir/data") is None
    assert PROBE.copy_local(environment, context, "/app", "data/input.txt /app/input.txt") is None

    assert environment.mounts == [((context / "data").resolve(), "/workdir/data")]
    assert environment.files[0][:2] == ("/app/input.txt", b"payload")


def test_execution_classification_prefers_observed_boundaries() -> None:
    boundary = PROBE.Action(
        phase="solution",
        label="solution/solve.sh",
        returncode=127,
        stop_reason=None,
        unsupported=("openssl",),
        unsupported_commands=("openssl",),
        commands=("openssl",),
        stderr="openssl: not implemented in shellsim\n",
    )

    assert PROBE.classify([boundary], 1, None) == "explicit_boundary"
    assert PROBE.classify([boundary], 0, None) == "passed_with_boundary"
    assert PROBE.classify([boundary], 0, "0") == "boundary_and_verifier_failed"
    assert PROBE.classify([], 1, None) == "verifier_failed"
    assert PROBE.classify([], 0, "1") == "passed"
    assert PROBE.classify([], 0, "0.5") == "partial_reward"
    assert PROBE.classify([], 0, "0") == "verifier_failed"


def test_provisioning_detection_covers_assignment_prefixes_and_package_managers() -> None:
    provisioning = [
        "DEBIAN_FRONTEND=noninteractive apt-get install -y curl",
        "apk add bash",
        "python3 -m pip install numpy",
        "cargo install ripgrep",
        "uv sync",
    ]

    assert all(PROBE.is_image_provisioning(source) for source in provisioning)
    assert PROBE.is_image_provisioning("<<EOF\necho generated\nEOF")
    assert not PROBE.is_image_provisioning("python generate_data.py")


def test_docker_environment_supports_assignment_and_legacy_forms() -> None:
    assert PROBE.docker_environment("PATH=/bin MODE=test") == {"PATH": "/bin", "MODE": "test"}
    assert PROBE.docker_environment("MESSAGE hello world") == {"MESSAGE": "hello world"}
    assert PROBE.expand_docker_environment("$ROOT/app", {"ROOT": "/work"}) == "/work/app"
    assert PROBE.expand_docker_environment("$PATH:/app", {"PATH": "/bin"}) == "/bin:/app"


def test_replay_runs_solution_then_verifier_in_one_environment(tmp_path: Path) -> None:
    task = tmp_path / "sample"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.13\nWORKDIR /app\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("printf solved > result.txt\n")
    (task / "tests").mkdir()
    (task / "tests" / "test_result.py").write_text("def test_result():\n    assert True\n")

    class Result:
        returncode = 0
        stop_reason = None
        unsupported: tuple[str, ...] = ()
        unsupported_commands: tuple[str, ...] = ()
        commands: tuple[str, ...] = ()
        stderr_text = ""

    class Environment(FakeEnvironment):
        instances: list["Environment"] = []

        def __init__(self, _limits: object) -> None:
            super().__init__()
            self.terminated = False
            self.sources: list[str] = []
            self.instances.append(self)

        def run(self, source: str) -> Result:
            self.sources.append(source)
            return Result()

        def read_file(self, _path: str) -> bytes:
            raise RuntimeError("absent")

    options = SimpleNamespace(cpu=1, memory=2, disk=3, output=4)
    result = PROBE.replay_task(task, Environment, lambda **values: values, options)

    assert len(Environment.instances) == 1
    sources = Environment.instances[0].sources
    assert sources.index("cd /app && bash /solution/solve.sh") < sources.index(
        "cd /tests && pytest /tests/test_result.py"
    )
    assert result["category"] == "passed"
    assert result["solution_outcome"] == "passed"
    assert result["verifier_outcome"] == "passed"


def test_replay_keeps_wrapper_boundary_separate_from_normalized_result(tmp_path: Path) -> None:
    task = tmp_path / "sample"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.13\nWORKDIR /app\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("true\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("apt-get install pytest\n")
    (task / "tests" / "test_result.py").write_text("def test_result():\n    assert True\n")

    class Result:
        returncode = 0
        stop_reason = None
        unsupported: tuple[str, ...] = ()
        unsupported_commands: tuple[str, ...] = ()
        commands: tuple[str, ...] = ()
        stderr_text = ""

    class WrapperBoundary(Result):
        returncode = 127
        unsupported = ("apt-get",)
        unsupported_commands = ("apt-get",)
        commands = ("apt-get",)
        stderr_text = "apt-get: not implemented in shellsim\n"

    class Environment(FakeEnvironment):
        def __init__(self, _limits: object) -> None:
            super().__init__()
            self.terminated = False

        def run(self, source: str) -> Result:
            return WrapperBoundary() if source == "cd /tests && bash /tests/test.sh" else Result()

        def read_file(self, _path: str) -> bytes:
            raise RuntimeError("absent")

    options = SimpleNamespace(cpu=1, memory=2, disk=3, output=4)
    result = PROBE.replay_task(task, Environment, lambda **values: values, options)

    assert result["category"] == "passed"
    assert result["verifier_source"] == "normalized_payload"
    assert result["first_boundary"] is None
    assert result["first_wrapper_boundary"]["unsupported"] == ["apt-get"]
