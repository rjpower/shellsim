"""Exercise the Dockerfile subset through its simulated filesystem effects."""

import shlex
from pathlib import Path

import pytest
from shellsim import Environment, SimulationError

from tools.tasktrove_runtime_probe import apply_docker_layout, docker_layout


def test_apply_docker_layout_overlays_declared_destinations_and_modes(tmp_path: Path) -> None:
    task = tmp_path / "task"
    environment = task / "environment"
    (environment / "data").mkdir(parents=True)
    (environment / "data" / "value.txt").write_text("copied\n")
    (environment / "script.sh").write_text("#!/bin/sh\nprintf ready\n")
    (environment / "Dockerfile").write_text(
        "FROM ignored\nWORKDIR /app\nCOPY data/ relative/\nCOPY script.sh /usr/local/bin/\n"
        "COPY --from=builder data/ /app/generated\n"
        "RUN apt-get update \\\n  && chmod +x /usr/local/bin/script.sh\nWORKDIR /app/relative\n"
    )
    simulated = Environment()

    layout = docker_layout(task)
    apply_docker_layout(simulated, task, layout)

    result = simulated.run(f"cd {shlex.quote(layout.workdir)} && pwd && cat value.txt && /usr/local/bin/script.sh")
    assert result.returncode == 0, result.stderr_text
    assert result.stdout == b"/app/relative\ncopied\nready"
    with pytest.raises(SimulationError):
        simulated.read_file("/app/generated/value.txt")
