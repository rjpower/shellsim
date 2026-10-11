"""Linux process identities, owned group cleanup and workspace accounting.

These helpers are shared by the worker client and detached supervisor without
importing the executable runner module during package initialization.
"""

from __future__ import annotations

import os
import signal
import stat
import time
from pathlib import Path


def process_token(pid: int) -> str | None:
    """Linux boot/start identity prevents signalling a reused supervisor PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return boot + ":" + fields[19]
    except FileNotFoundError:
        return None


def kill_group(pid: int) -> None:
    """The supervisor owns the group until terminal status has been written."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def wait_group(pid: int) -> None:
    """Wait for killed descendants as well as the direct child before sealing."""
    while True:
        alive = False
        with os.scandir("/proc") as processes:
            for process in processes:
                if not process.name.isdecimal():
                    continue
                try:
                    fields = Path(process.path, "stat").read_text().rsplit(")", 1)[1].split()
                except FileNotFoundError:
                    continue
                if int(fields[2]) == pid and fields[0] != "Z":
                    alive = True
                    break
        if not alive:
            return
        time.sleep(0.01)


def tree_usage(root: Path, max_files: int, max_bytes: int, *, skip_inputs: bool = False) -> tuple[int, int]:
    """Bound workspaces without following links or materializing directory lists."""
    directories = [root]
    count = size = 0
    while directories:
        directory = directories.pop()
        try:
            children = os.scandir(directory)
        except FileNotFoundError:
            if directory == root:
                raise
            continue
        with children:
            for child in children:
                if skip_inputs and directory == root and child.name == "inputs":
                    continue
                try:
                    info = child.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                count += 1
                if stat.S_ISDIR(info.st_mode):
                    directories.append(Path(child.path))
                elif stat.S_ISREG(info.st_mode):
                    size += info.st_size
                if count > max_files or size > max_bytes:
                    raise ValueError("workspace exceeds resource bounds")
    return count, size
