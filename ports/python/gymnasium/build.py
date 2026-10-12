"""Preserve the official wheel while deferring its unsupported process backend."""

from pathlib import Path
from tempfile import TemporaryDirectory

from ports._support.pure_wheel import verified_files
from ports._support.python_adapters import PythonBuildOutput, _wheel
from ports.api import BuildContext


def build(ctx: BuildContext):
    files = verified_files(ctx.source, dict(ctx.metadata))
    name = "gymnasium/vector/__init__.py"
    eager = b"from gymnasium.vector.async_vector_env import AsyncVectorEnv\n"
    if files[name].count(eager) != 1:
        raise ValueError("Gymnasium's pinned async import differs")
    files[name] = (
        files[name].replace(eager, b"", 1)
        + b"""

def __getattr__(name):
    if name == "AsyncVectorEnv":
        from gymnasium.vector.async_vector_env import AsyncVectorEnv

        return AsyncVectorEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
"""
    )
    record = f"gymnasium-{ctx.port.version}.dist-info/RECORD"
    wheels = ctx.result / "wheels"
    wheels.mkdir(parents=True)
    with TemporaryDirectory(prefix=".gymnasium-wheel-", dir=ctx.result) as directory:
        stage = Path(directory)
        for name, data in files.items():
            if name == record:
                continue
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        _wheel(stage, wheels / ctx.source.name, record)
    return PythonBuildOutput(ctx.result, ())
