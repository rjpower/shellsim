"""Atomic local cache admission without exposing interrupted transfers."""

import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def staged_file(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as source:
        temporary = Path(source.name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_bytes(path: Path, value: bytes) -> None:
    with staged_file(path) as temporary:
        temporary.write_bytes(value)
