"""Direct execution of source by shellsim's isolated Python implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

from ._api import Environment, Limits, RunResult


def run(
    source: str,
    argv: Sequence[str] = (),
    stdin: Union[bytes, bytearray, memoryview] = b"",
    limits: Optional[Limits] = None,
    *,
    cpu: Optional[int] = None,
    memory: Optional[int] = None,
    disk: Optional[int] = None,
    output: Optional[int] = None,
) -> RunResult:
    """Execute Python source in a fresh, resource-bounded shellsim environment."""

    return Environment(
        limits,
        cpu=cpu,
        memory=memory,
        disk=disk,
        output=output,
    ).run_python(source, argv, stdin)
