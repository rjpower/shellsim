"""Bounded source extraction and command logging shared by SDK producers."""

import hashlib
import resource
import subprocess
import tarfile
from pathlib import Path

from ports._support.store import file_hash


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(command, log, environment):
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    with log.open("w") as output:
        subprocess.run(
            command,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3600,
            preexec_fn=limit,
        )


def extract(archive, destination, digest, selected):
    if file_hash(archive) != digest:
        raise ValueError("toolchain source archive SHA-256 mismatch")
    if destination.exists():
        raise ValueError("use a fresh toolchain work directory")
    destination.mkdir()
    with tarfile.open(archive) as source:
        members = []
        for member in source:
            parts = Path(member.name).parts
            if len(parts) < 2 or (selected and parts[1] not in selected):
                continue
            member.name = str(Path(*parts[1:]))
            if member.islnk():
                member.linkname = str(Path(*Path(member.linkname).parts[1:]))
            members.append(member)
        source.extractall(destination, members=members, filter="data")
