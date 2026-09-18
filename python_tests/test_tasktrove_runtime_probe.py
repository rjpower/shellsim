"""Tests for the bounded TaskTrove prioritization probe's Dockerfile subset."""

from pathlib import Path

from shellsim import Environment
from tools.tasktrove_runtime_probe import (
    DockerCopy,
    DockerLayout,
    apply_docker_layout,
    docker_instructions,
    docker_layout,
)


def test_docker_instructions_join_continuations_without_shell_execution() -> None:
    assert docker_instructions("RUN apt-get update \\\n  && apt-get install awk\nCOPY data /work/data\n") == (
        "RUN apt-get update && apt-get install awk",
        "COPY data /work/data",
    )


def test_docker_layout_tracks_workdirs_literal_copies_and_chmod(tmp_path: Path) -> None:
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(
        """
        FROM ignored
        WORKDIR /app
        COPY data/ relative/
        COPY script.sh /usr/local/bin/
        COPY --from=builder /tmp/generated /app/generated
        RUN apt-get update && chmod +x /usr/local/bin/script.sh
        WORKDIR /app/relative
        """
    )

    assert docker_layout(environment.parent) == DockerLayout(
        workdir="/app/relative",
        copies=(
            DockerCopy(("data/",), "/app/relative", True),
            DockerCopy(("script.sh",), "/usr/local/bin", True),
        ),
        chmod_commands=("chmod +x /usr/local/bin/script.sh",),
    )


def test_apply_docker_layout_overlays_declared_destinations_and_modes(tmp_path: Path) -> None:
    task = tmp_path / "task"
    environment = task / "environment"
    (environment / "data").mkdir(parents=True)
    (environment / "data" / "value.txt").write_text("copied\n")
    (environment / "script.sh").write_text("#!/bin/sh\nprintf ready\n")
    (environment / "Dockerfile").write_text(
        "WORKDIR /app\nCOPY data/ /seed/\nCOPY script.sh /usr/local/bin/\nRUN chmod +x /usr/local/bin/script.sh\n"
    )
    simulated = Environment()

    layout = docker_layout(task)
    apply_docker_layout(simulated, task, layout)

    result = simulated.run("cat /seed/value.txt; /usr/local/bin/script.sh")
    assert result.returncode == 0, result.stderr_text
    assert result.stdout == b"copied\nready"
